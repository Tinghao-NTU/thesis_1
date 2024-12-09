import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class policy_net(nn.Module):
    def __init__(self, state_size,output_size, layer_type='bbb', activation_type='softplus'):
        super(policy_net, self).__init__() 
        
        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.active = nn.Softplus
        elif activation_type=='relu':
            self.active = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")
            
        self.n_hidden = 256
        self.num_layers = 2
#         self.GRU = nn.GRU(state_size, 256,2,batch_first = True)
        self.fc1 = BBBLinear(state_size, 128)
        self.ac1 = self.active()
        
        self.fc2 = BBBLinear(128,128)
        self.ac2 = self.active()

        self.fc3 = BBBLinear(128,128)
        self.ac3 = self.active()
        
        self.out = BBBLinear(128,output_size)
        
    def forward(self, x): 
        if str(type(x)) != "<class 'torch.Tensor'>":
            x = torch.from_numpy(x).float()
        x = x.cuda()
        x = self.ac1(self.fc1(x))
        x = self.ac2(self.fc2(x))
        x = self.ac3(self.fc3(x))
        action_prob = self.out(x)

        return action_prob.cpu()

    def act(self, input,explore = True):
#         if len(np.shape((input))) <3:
#             input = np.expand_dims(input, 0)
        prob = self.forward(input)
        if explore:
            action = gumbel_softmax(prob, hard=True)
        else:
            action = onehot_from_logits(prob)
        return action



class value_net(nn.Module):
    def __init__(self, input1_dim, input2_dim, output_dim, layer_type='lrt', activation_type='relu'):
        super(value_net, self).__init__()
        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.active = nn.Softplus
        elif activation_type=='relu':
            self.active = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")        
     
        
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.output_dim = output_dim

        self.fc1 = BBBLinear(self.input1_dim + self.input2_dim, 512)
        self.ac1 = self.active()
        
        self.fc2 = BBBLinear(512,512)
        self.ac2 = self.active()

        self.fc3 = BBBLinear(512,512)
        self.ac3 = self.active()
        
        self.out = BBBLinear(512,self.output_dim)



    def forward(self, input1, input2):
        x = torch.cat([input1, input2], 1)
        if str(type(x)) != "<class 'torch.Tensor'>":
            x = torch.from_numpy(x).float()
        x = x.cuda()
        x = self.ac1(self.fc1(x))
        x = self.ac2(self.fc2(x))
        x = self.ac3(self.fc3(x))
        action_prob = self.out(x)
        
        return action_prob
   


def gumbel_softmax(logits, temperature=2.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = tens_type(*shape).uniform_()
    return -torch.log(-torch.log(U + eps) + eps)
def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]]
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])
# class policy_net(nn.Module):
#     def __init__(self, state_size,output_size):
#         super(policy_net, self).__init__()        
#         self.n_hidden = 256
#         self.num_layers = 2
# #         self.GRU = nn.GRU(state_size, 256,2,batch_first = True)
#         self.advantage = nn.Sequential(
#             nn.Linear(state_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),            
#             nn.Linear(512, output_size)
#         )
        
#     def forward(self, x): 
#         if str(type(x)) != "<class 'torch.Tensor'>":
#             x = torch.from_numpy(x).float()
#         outputs = self.advantage(x)
# #         outputs = self.advantage(self.GRU(x)[0])
# #         outputs = outputs.view(x.size()[0],-1)
#         return outputs

#     def act(self, input,explore = True):
# #         if len(np.shape((input))) <3:
# #             input = np.expand_dims(input, 0)
#         prob = self.forward(input)
#         if explore:
#             action = gumbel_softmax(prob, hard=True)
#         else:
#             action = onehot_from_logits(prob)
#         return action


# class value_net(nn.Module):
#     def __init__(self, input1_dim, input2_dim, output_dim):
#         super(value_net, self).__init__()
#         self.input1_dim = input1_dim
#         self.input2_dim = input2_dim
#         self.output_dim = output_dim

#         self.fc1 = nn.Linear(self.input1_dim + self.input2_dim, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, self.output_dim)

#         gain = nn.init.calculate_gain('leaky_relu')
#         nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
#         nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
#         nn.init.xavier_uniform_(self.fc3.weight, gain=gain)

#     def forward(self, input1, input2):
#         x = torch.cat([input1, input2], 1)
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = self.fc3(x)
#         return x