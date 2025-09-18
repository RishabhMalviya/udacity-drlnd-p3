import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size=24, action_size=2, seed=42, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self._reset_parameters()
        
    def _reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size=24, action_size=2, seed=42, fcs1_units=192, fca1_units=64, fc2_units=128):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fca1 = nn.Linear(action_size, fca1_units)
        self.fc2 = nn.Linear(fcs1_units + fca1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fca1.weight.data.uniform_(*hidden_init(self.fca1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        xa = F.relu(self.fca1(action))

        x = F.relu(self.fc2(torch.cat((xs, xa), dim=1)))

        return self.fc3(x)
