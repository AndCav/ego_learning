import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def weights_init_uniform(m):
    if isinstance(m, nn.BatchNorm1d):
        torch.nn.init.xavier_uniform(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0.1)


class acNetCell(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, gru_hidden_size, iso):
        super(acNetCell, self).__init__()

        self.iso = iso
        self.linear = nn.Linear(num_inputs, hidden_size[0])
        self.hidden = []
        for i in range(len(hidden_size) - 1):
            self.hidden.append(nn.Linear(hidden_size[i], hidden_size[i + 1]).cuda())
            # self.hidden.append(nn.BatchNorm1d(hidden_size[i + 1]))
            # self.hidden.append(nn.ReLU())

        self.gru = nn.GRU(hidden_size[-1], gru_hidden_size, batch_first=False)
        self.critic_linear = nn.Sequential(nn.Linear(gru_hidden_size, 100), nn.ReLU(), nn.Linear(100, 1))
        self.actor_linear = nn.Linear(gru_hidden_size, 100)
        self.mu_linear = nn.Linear(100, num_actions)

        if self.iso:
            self.sigma_linear = torch.nn.Parameter(torch.zeros(1, num_actions))
        else:
            self.sigma_linear = nn.Linear(100, num_actions)

    def forward(self, x, hx):

        x = torch.tanh(self.linear(x))

        for layer in self.hidden:
            x = torch.tanh(layer(x))

        # nn.utils.rnn.pack_padded_sequence(x, None, batch_first=False)

        x, hx = self.gru(x, hx)
        # x,_ = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
        actor = torch.tanh(self.actor_linear(x))
        mu = self.mu_linear(actor)
        if self.iso:
            sigma = self.sigma_linear
        else:
            sigma = self.sigma_linear(actor)
        return mu, sigma, self.critic_linear(x)

    def single_forward(self, x, hx):
        x = torch.tanh(self.linear(x))
        # print(hx.shape)
        # print(x.shape)

        for layer in self.hidden:
            # print(x)
            x = torch.tanh(layer(x))

        x, hx = self.gru(x, hx)
        actor = torch.tanh(self.actor_linear(x))
        mu = self.mu_linear(actor)
        if self.iso:
            sigma = self.sigma_linear
        else:
            sigma = self.sigma_linear(actor)
        return mu, sigma, self.critic_linear(x), hx


def save_checkpoint(save_path, episode, model, optimizer, obs):
    if save_path == None:
        return
    save_path = '%s/%d.pt' % (save_path, episode)
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'obs_state_dict': obs.save()}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')


def load_checkpoint(save_path, episode, model, optimizer, obs):
    save_path = '%s/%d.pt' % (save_path, episode)
    state_dict = torch.load(save_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    obs.load(state_dict['obs_state_dict'])

    print(f'Model loaded from <== {save_path}')


class Shared_obs_stats():
    def __init__(self, num_inputs, step):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()
        self.step = float(step)

    def observes(self, obs):
        # observation mean var updates
        x = obs.data.squeeze().mean(dim=0)
        self.n += self.step
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs))
        obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp((inputs - obs_mean) / obs_std, -5., 5.)

    def save(self):
        return (self.n, self.mean, self.mean_diff, self.var)

    def load(self, input):
        n, mean, mean_diff, var = input
        print(input)
        self.n += n
        self.mean += mean
        self.mean_diff += mean_diff
        self.var += var 
