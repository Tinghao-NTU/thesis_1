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
        self.LSTM = nn.LSTM(n_states, n_hidden,num_layers,batch_first = True)
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.linear1 = nn.Linear(n_hidden, n_output)
        
        self.linear = nn.Linear(n_hidden, 1)
        
        self._init_weights()
        
    def forward(self, input):
        hns = torch.zeros(self.num_layers,1,self.n_hidden).cuda()
        cns = torch.zeros(self.num_layers,1,self.n_hidden).cuda()
        outputs = []
        values = []
        for i in range(len(input)):
            x = input[i,:].view(1,1,-1)
            output, (hns, cns) = self.LSTM(x, (hns, cns))
            output_policy = F.softmax(self.linear1(output), dim= 2)
            outputs.append(output_policy)
            
            values.append(self.linear(output))
        return torch.cat(outputs),torch.cat(values)
    
    def _init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.normal_(m.bias, 0, 0.01)