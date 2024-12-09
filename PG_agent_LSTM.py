import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple, deque, defaultdict
from functools import partial
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
from ReplayBuffer import ReplayMemory, MemoryBuffer
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def convert_to_tensor(data, type="float"):
    if type=="long":
        return torch.from_numpy(data).long().to(device)    
    else:
        return torch.from_numpy(data).float().to(device)
class MARL_QNetwork(nn.Module):
  # Defines the Q Network used for local and target networks
  def __init__(self, state_size):
    super(MARL_QNetwork, self).__init__()
    self.n_hidden = 256
    self.num_layers = 2
    self.GRU = nn.GRU(state_size, 128,2,batch_first = True, bidirectional  = True)
    self.advantage = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
        
  def forward(self, x): 
    outputs = F.softmax(self.advantage(F.relu(self.GRU(x)[0])),dim=1)#self.advantage(torch.tanh(self.GRU(x)[0]))#
#     outputs = F.softmax(self.advantage(F.relu(self.GRU(x)[0])),dim=1)
#     outputs = F.softmax(self.advantage(self.GRU(x)[0]),dim=1)
    return outputs.view(x.size()[0],-1)[:,:-1]


class MARL_Agent(object):
  def __init__(self, state_size, action_size, seed,
               update_every=5, lr=1e-3,
               buffer_size = 1000, batch_size=5,
               gamma = 0.99, tau = 1e-1):
    super(MARL_Agent, self).__init__()
    # Model settings
    self.state_size = state_size
    self.action_size = action_size
    self.actions = np.arange(self.action_size)
    # Model parameters
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.gamma=gamma
    self.tau = tau
    self.update_step = update_every
    # Model seed
    self.seed = random.seed(seed)

    # Debugging variables
      
    #Q- Network
    self.Net_policy = MARL_QNetwork(state_size).to(device)


    self.log_act_prob = []
    # Training setup
    self.optimizer = optim.Adam(self.Net_policy.parameters(),lr=lr)
    self.lstep = 0
    self.policy_loss_total = []


  def act(self, state, name = None, show = None):
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    action_probs = self.Net_policy(state_tensor)
    m = Categorical(action_probs)
    action = m.sample().cpu().numpy().item()
    m_log_prob = m.log_prob(torch.tensor(action).to(device))
    self.log_act_prob.append(m_log_prob)
    if name is None:
        name = 'probs_GRU_mnist_re.txt'
    file = open(name,mode='a')

    file.write(str(action_probs)+'\n')

    file.close()

    if show:
        print(action_probs)
    return action



  def store(self, policy_reward): 
    policy_loss = []
    R = 0
    returns = []
    eps = np.finfo(np.float32).eps.item()
    for r in policy_reward[::-1]:
        R = r + 0.9 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(self.log_act_prob, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    self.policy_loss_total.append(policy_loss)
#     policy_loss.backward()
#     self.optimizer.step()    
#     self.log_act_prob = []
  def learn(self): 
    self.policy_loss_total = sum(self.policy_loss_total)
    self.optimizer.zero_grad()
    self.policy_loss_total.backward()
    self.optimizer.step()
    self.policy_loss_total = []
    self.log_act_prob = []