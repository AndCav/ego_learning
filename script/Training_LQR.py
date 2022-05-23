import os
import sys
import gym
from gym import wrappers
import random
import numpy as np
import math
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ACNet import save_checkpoint, load_checkpoint


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.load = 0
        self.batchsize = 0
        self.memory = []

    def push(self, events):
        # Events = list(zip(*events))
        # self.memory.append(map(lambda x: torch.cat([torch.repeat_interleave(torch.zeros_like(x[0]),(1000-len(x)),dim = 0),
        # torch.cat(x, 0)],0), events))
        self.memory.append(map(lambda x: torch.cat(x, 0), events))
        self.load += len(events[1])
        self.batchsize += 1
        # if len(self.memory)>self.capacity:
        # del self.memory[0]

    def clear(self):
        self.memory = []
        self.load = 0
        self.batchsize = 0

    def pull(self):
        Memory = list(zip(*self.memory))
        data = map(lambda x: torch.cat(x, 1), Memory)
        return data


def train(env, model, optimizer, shared_obs_stats, device, params):
    memory = ReplayMemory(params.num_steps)
    state = env.reset()
    done = True
    episode = params.initial_model
    n = params.robot_number
    # model.train()
    env_action = None
    action = None
    # horizon loop
    for t in range(params.time_horizon):
        model.eval()
        episode_length = 0
        while (memory.load < params.num_steps * params.batch_size * n):
            states = []
            actions = []
            rewards = []
            values = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0
            xdis = []
            for i in range(n):
                xdis.append(0)
            hx = torch.zeros((n, params.gruhiddensize)).unsqueeze(0).to(device)

            # n steps loops
            step = 0
            while step in range(params.num_steps):

                state = Variable(torch.Tensor(state))
                episode_length += 1
                state = state.reshape(n, 12)
                shared_obs_stats.observes(state)
                state = shared_obs_stats.normalize(
                    state).unsqueeze(0).to(device)
                states.append(state)
                # print(hx.shape)
                # print(state.shape)
                model = model.to(device)
                with torch.no_grad():
                    mu, sigma, v, hx = model.single_forward(state, hx)
                # h.append(hx)
                # c.append(cx)
                # print(v.shape)

                action = (mu + torch.exp(sigma)*Variable(torch.randn(mu.size()).to(device)))
                actions.append(action)
                log_prob = -0.5 * ((action - mu) / torch.exp(sigma)
                                   ).pow(2) - (0.5 * math.log(2 * math.pi)) - sigma
                log_prob = log_prob.sum(-1, keepdim=True)
                logprobs.append(log_prob)
                values.append(v)
                action = action.reshape(1, n * 6)
                env_action = torch.tanh(
                    action).data.squeeze().cpu().numpy()
                state, reward, done, _ = env.step(env_action)
                # if(reward <= -10000):
                #     step = params.num_steps
                #     done = True
                # if step % 128 == 0:
                # for i in range(n):
                # xdis[i] = state[60 * i + 48] - xdis[i]
                # reward[i] += xdis[i] * 80
                cum_reward += reward
                # reward = max(min(reward, 1), -1)
                rewards.append(reward)

                if done:
                    episode += 1
                    cum_done += n
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    with torch.no_grad():
                        state = Variable(torch.Tensor(state))
                        state = state.reshape(n, 12)
                        shared_obs_stats.observes(state)
                        state = shared_obs_stats.normalize(
                            state).unsqueeze(0).to(device)
                        _, _, v, _ = model.single_forward(state, hx)
                        values.append(v)
                    state = env.reset()
                    break
                step = step+1

            # compute returns and GAE(lambda) advantages:
            for j in range(n):
                returns = []
                advantages = []
                st = []
                ac = []
                lo = []
                R = torch.zeros(1, 1)
                A = Variable(torch.zeros(1, 1)).to(device)
                for i in reversed(range(len(rewards))):
                    td = rewards[i] + params.gamma * \
                        values[i + 1].data[0, j] - values[i].data[0, j]
                    A = float(td) + params.gamma * params.gae_param * A
                    advantages.insert(0, A)
                    R = A + values[i][0][j]
                    returns.insert(0, R)
                    st.insert(0, states[i][0][j].unsqueeze(0).unsqueeze(0))
                    ac.insert(0, actions[i][0][j].unsqueeze(0).unsqueeze(0))
                    lo.insert(0, logprobs[i][0][j].unsqueeze(0).unsqueeze(0))
                # print(advantages[0].shape) torch.Size([1, 1])
                memory.push([st, ac, returns, advantages, lo])

        model.train()
        # print(memory.load)
        # epochs
        batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = memory.pull()
        batch_advantages = batch_advantages.unsqueeze(-1)
        batch_returns = batch_returns.unsqueeze(-1)
        # print(batch_states.shape)torch.Size([1024, 4, 60])
        for k in range(params.num_epoch):
            # new probas
            hx = torch.zeros(
                (memory.batchsize, params.gruhiddensize)).unsqueeze(0).to(device)

            # size: length * batch * sigma_size
            Mu, Sigma, V_pred = model(batch_states, hx)
            # Sigma = Sigma.expand_as(Mu)

            log_probs = -0.5 * ((batch_actions - Mu) / torch.exp(Sigma)
                                ).pow(2) - 0.5 * math.log(2 * math.pi) - Sigma
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + Sigma
            dist_entropy = dist_entropy.mean().sum(-1, keepdim=True)

            # ratio
            ratio = torch.exp(log_probs - batch_logprobs)

            # clip loss
            # print(ratio.shape)
            # print(batch_advantages.shape)

            # surrogate from conservative policy iteration
            surr1 = ratio * batch_advantages.expand_as(ratio)
            surr2 = ratio.clamp(1 - params.clip, 1 + params.clip) * \
                batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))

            # value loss
            loss_value = (V_pred - batch_returns).pow(2).mean()

            # entropy
            loss_ent = - params.ent_coeff * dist_entropy

            # gradient descent step
            total_loss = (loss_clip + loss_value + loss_ent)

            # loss = torch.square(log_probs - torch.full_like(batch_logprobs, 0)).mean() + torch.square(
            # V_pred - torch.full_like(batch_returns, 1.0)).mean()

            # print(loss)
            optimizer.zero_grad()
            total_loss.backward(retain_graph=params.iso_sig)
            # loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), params.max_grad_norm)
            optimizer.step()

        # finish, print:
         # av_reward = sum(av_reward) #only for multiple robot
        print('episode', episode, 'av_reward', av_reward /
              float(cum_done), 'total loss', total_loss)
        if episode % params.save_interval == 0:
            save_checkpoint(params.save_path, episode, model,
                            optimizer, shared_obs_stats)
            f = open(params.save_path + ".txt", "a")
            f.write("%f %f\n" % (av_reward / float(cum_done), total_loss))
            f.close()
        memory.clear()
        # print("free cache")
        # torch.cuda.empty_cache() 


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def readtxt(name):
    mat = np.loadtxt(name, dtype='f', delimiter=' ')
    return mat
