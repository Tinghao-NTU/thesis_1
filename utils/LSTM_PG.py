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
    def __init__(self,n_states, n_hidden, n_output,num_layers = 2):
        super(Policy, self).__init__()
        self.LSTM = nn.LSTM(n_states, n_hidden,num_layers)
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.linear = nn.Linear(n_hidden, n_output)
    def forward(self, input):
        hns = torch.zeros(self.num_layers,1,self.n_hidden).cuda()
        cns = torch.zeros(self.num_layers,1,self.n_hidden).cuda()
        outputs = []
        for i in range(len(input)):
            x = input[i,:].view(1,1,-1)
            output, (hns, cns) = self.LSTM(x, (hns, cns))
            output = F.softmax(self.linear(output), dim= 2)
            outputs.append(output)
        return torch.cat(outputs)