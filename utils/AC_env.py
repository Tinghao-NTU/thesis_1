import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

class Policy(nn.Module):
    def __init__(self,n_states, n_hidden, n_output, init_w=0.05):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_output)
       
        self.linear = nn.Linear(n_hidden, 1)
 #这是policy的参数
        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []
#这是state_action_func的参数
        # self.Reward = []
        # self.s_value = []
        self._init_parameters()
    def forward(self, x):
        x = F.relu(self.linear2(F.relu(self.linear1(x))))
        output = F.softmax(self.linear3(x),dim = 0)
        
        state_values = self.linear(x)
        return output.view(-1), state_values
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.normal_(m.bias, 0, 0.01)