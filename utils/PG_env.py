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
    def __init__(self,n_states, n_hidden, n_output):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_output)

 #这是policy的参数
        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []
#这是state_action_func的参数
        # self.Reward = []
        # self.s_value = []

    def forward(self, x):
        x = F.relu(self.linear2(F.relu(self.linear1(x))))
        output =self.linear3(x)
        output = F.softmax(output,dim = 1)
        return output.view(-1)

class Policy_LSTM(nn.Module):
    def __init__(self,n_states, n_hidden, n_output):
        super(Policy_LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.num_layers = 2

        self.LSTM = nn.LSTM(n_states, self.n_hidden,2,batch_first = True)

        
        self.advantage = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 1)
        )



    def forward(self, x):
        hns = torch.zeros(self.num_layers,x.size()[0],self.n_hidden).cuda()
        cns = torch.zeros(self.num_layers,x.size()[0],self.n_hidden).cuda()
        outputs = []

        for i in range(x.size()[1]):
#             print(np.shape(x))
            input = x.view(x.size()[0],x.size()[1],10)[:,i,:].view(-1,1,10).float()
            output, (hns, cns) = self.LSTM(input, (hns, cns))
            output_policy = self.advantage(output)
#             print(np.shape(output_policy))
            outputs.append(output_policy)


        advantage = F.softmax(torch.cat(outputs).view(x.size()[0],-1),dim = 1)

        return advantage.view(-1)