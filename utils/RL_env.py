import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

env_id = "Federated-v0"
env = gym.make(env_id)
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state.cpu(), 0)
        next_state = np.expand_dims(next_state.cpu(), 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQN, self).__init__()
        
        
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
       
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            #state   = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            #action  = q_value.max(1)[1].data[0]
            action  = torch.sort(q_value)[1].view(-1)[-10:].cpu().detach().numpy()
        else:
            action  = np.random.choice(range(100), 10, replace=False)
            #action = max(index)
            
        return action

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())    

def compute_td_loss(batch_size,replay_buffer,current_model,target_model,gamma,optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.from_numpy(state).cuda()
    next_state = torch.from_numpy(next_state).cuda()
    action     = torch.LongTensor(action).cuda()
    reward     = torch.FloatTensor(reward).cuda()
    done       = torch.FloatTensor(done).cuda().view(batch_size,-1)

    q_values      = current_model(state)
    next_q_values = target_model(next_state)

    q_value          = q_values[0,action]#q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = torch.sort(next_q_values)[0].view(-1)[-10:] #next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def plot_RL(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()