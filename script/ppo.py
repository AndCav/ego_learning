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

from model import Model, Shared_obs_stats
import ego_learning.register_all_env
import rospy


class Params():
    def __init__(self):
        self.batch_size = 16
        self.lr = 7e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.01
        self.num_epoch = 16
        self.num_steps = 512
        self.time_horizon = 2000
        self.max_episode_length = 2000
        self.saving_interval = 16
        self.max_grad_norm = 0.5
        self.seed = 1
        self.env_name = 'egoStandupEnv_LQR-v0'
        self.mode = 'load'
        self.load_path = 'net-backup.pt'
        self.save_path = '~/training_results'


def save_checkpoint(save_path, episode, model, optimizer, obs):
    save_path = '%s/%d.pt' % (save_path, episode)
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'obs_state_dict': obs.save()}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')


def load_checkpoint(file_name, model, optimizer):
    save_path = file_name
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    print(f'Model loaded from <== {save_path}')


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


def train(env, model, optimizer, shared_obs_stats, device):
    memory = ReplayMemory(params.num_steps)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    done = True
    episode = 0
    if not (os.path.exists(params.save_path)):
        print("saving folder")
        os.makedirs(params.save_path)

    if not (os.path.exists(params.save_path)):
        os.makedirs(params.save_path)

    # horizon loop
    for t in range(params.time_horizon):
        episode_length = 0
        while(memory.load < params.num_steps * params.batch_size):  # non esce mai da qua dentro
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0

            # n steps loops
            for step in range(params.num_steps):
                episode_length += 1
                shared_obs_stats.observes(state)
                state = shared_obs_stats.normalize(state)
                states.append(state)
                mu, sigma_sq, v = model(state)
                action = (mu + sigma_sq*Variable(torch.randn(mu.size())))
                actions.append(action)
                log_std = model.log_std
                log_prob = -0.5 * ((action - mu) / sigma_sq).pow(2) - \
                    0.5 * math.log(2 * math.pi) - log_std
                log_prob = log_prob.sum(-1, keepdim=True)
                logprobs.append(log_prob)
                values.append(v)
                env_action = action.data.squeeze().numpy()
                state, reward, done, _ = env.step(env_action)
                done = (done or episode_length >= params.max_episode_length)
                cum_reward += reward
                # reward = max(min(reward, 1), -1)
                rewards.append(reward)

                if done:
                    episode += 1
                    cum_done += 1
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    with torch.no_grad():
                        state = Variable(torch.Tensor(state))
                        state = state.reshape(1, 12)
                        shared_obs_stats.observes(state)
                        state = shared_obs_stats.normalize(
                            state).unsqueeze(0).to(device)
                        _, _, v, _ = model.forward(state, hx)
                        values.append(v)
                    state = env.reset()
                    break

            # one last step
            R = torch.zeros(1, 1)
            if not done:
                _, _, v = model(state)
                R = v.data

            # compute returns and GAE(lambda) advantages:
            R = Variable(R)
            values.append(R)
            A = Variable(torch.zeros(1, 1))
            for i in reversed(range(len(rewards))):
                td = rewards[i] + params.gamma * \
                    values[i+1].data[0, 0] - values[i].data[0, 0]
                A = float(td) + params.gamma*params.gae_param*A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            # store usefull info:
            memory.push([states, actions, returns, advantages, logprobs])

        # epochs
        for k in range(params.num_epoch):
            batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = memory.sample(
                params.batch_size)
            batch_actions = Variable(batch_actions.data, requires_grad=False)
            batch_states = Variable(batch_states.data, requires_grad=False)
            batch_returns = Variable(batch_returns.data, requires_grad=False)
            batch_advantages = Variable(
                batch_advantages.data, requires_grad=False)
            batch_logprobs = Variable(batch_logprobs.data, requires_grad=False)

            # new probas
            mu, sigma_sq, v_pred = model(batch_states)
            log_std = model.log_std
            log_probs = -0.5 * \
                ((batch_actions - mu) / sigma_sq).pow(2) - \
                0.5 * math.log(2 * math.pi) - log_std
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
            dist_entropy = dist_entropy.sum(-1).mean()

            # ratio
            ratio = torch.exp(log_probs - batch_logprobs)

            # clip loss
            # surrogate from conservative policy iteration
            surr1 = ratio * batch_advantages.expand_as(ratio)
            surr2 = ratio.clamp(1-params.clip, 1+params.clip) * \
                batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))

            # value loss
            loss_value = (v_pred - batch_returns).pow(2).mean()

            # entropy
            loss_ent = - params.ent_coeff * dist_entropy

            # gradient descent step
            total_loss = (loss_clip + loss_value + loss_ent)
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), params.max_grad_norm)
            optimizer.step()

        # finish, print:
        # 15s 4 episodes
        # if (episode) % (saving_interval) == 0:
        #     save_checkpoint('./training_results',episode, model, optimizer)
        # print('episode',episode,'av_reward',av_reward/float(cum_done))
        # memory.clear()

            print('episode', episode, 'av_reward', av_reward /
                  float(cum_done))
            # if episode % params.saving_interval == 0:
            print("saving")
            save_checkpoint(params.save_path, episode, model,optimizer, shared_obs_stats)
            f = open(params.save_path + ".txt", "a")
            f.write("%f %f\n" % (av_reward / float(cum_done), total_loss))
            f.close()
        memory.clear()


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    params = Params()

    device = torch.device(
        'cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
    rospy.init_node('ego_gyb_ppo', anonymous=True, log_level=rospy.INFO)
    torch.manual_seed(params.seed)
    work_dir = mkdir('exp', 'ppo')
    monitor_dir = mkdir(work_dir, 'monitor')
    env = gym.make(params.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = Model(num_inputs, num_outputs)
    shared_obs_stats = Shared_obs_stats(num_inputs)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    if params.mode == 'load':
        load_checkpoint(params.load_path, model, optimizer)
    train(env, model, optimizer, shared_obs_stats, device)
