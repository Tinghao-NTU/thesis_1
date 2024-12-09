import gym
import matplotlib.pyplot as plt
import io
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
# from collections import namedtuple, deque, defaultdict
# from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from models.Nets import MLP, CNNMnist, CNNCifar,CNNMnist_Compare,CNNCifar_Compare,weigth_init
from utils.options import args_parser
import numpy as np
import copy
import math
import random
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import ReplayBuffer
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from PG_agent_LSTM import MARL_Agent,MARL_QNetwork
from torchvision import datasets, transforms
from utils.update_device_LSTM_PG import local_update,get_state,permute_device
from models.Update import LocalUpdate
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
def feature_cal_norm1(wlist, wglobal):
    state_list = []
    for i in range(100):
        para_list = []
        for (_, parameters1),(_, parameters2) in zip(wlist[i].items(),wglobal.items()):
            para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 1,dim = 0).cpu().numpy().item())
        state_list.append(sum(para_list))
#     para_list = torch.cat(para_list)
    return np.array(state_list)
def get_action(clusteded_dic,state,i,num):
    i = '%d'%(i)
    device_idx = np.array(clusteded_dic[i])
    state_cluster =  state[device_idx]
    if num <= len(device_idx):
        action = np.argsort(state_cluster)[-num:]
        return device_idx[action]
    else:
        return device_idx
        
    

bias = '_08'

scaler_2 = StandardScaler()
def feature_selection(weights, trans, epoch):
    para_list = []
    for name, parameters in weights.items():
        para_list.append(parameters.view(-1))
    para_list = torch.cat(para_list).view(17,-1).detach().cpu().numpy()
    if epoch == 0:
        #print('first')
        trans.fit(para_list)
        result = trans.transform(para_list)
    else:
        result = trans.transform(para_list)
    return torch.from_numpy(result).reshape(1,-1).cuda(),trans

def feature_cal(wlist, wglobal):
    state_list = []
    for i in range(100):
        para_list = []
        for (_, parameters1),(_, parameters2) in zip(wlist[i].items(),wglobal.items()):
            para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 2,dim = 0).cpu().numpy().item())
        state_list.append(torch.tensor(para_list).view(1,-1))
#     para_list = torch.cat(para_list)
    return torch.cat(state_list,0)

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.dataset = 'cifar'
# load dataset and split users

args.iid = False

if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)


elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users= cifar_noniid(dataset_train, args.num_users,0.5)
num_frames = 500
gamma      = 0.99
preset_accuracy = torch.tensor([[55.0]])
all_rewards = [10000000]
episode_reward = 0
eps = np.finfo(np.float32).eps.item()

# with open('data' + bias +'.json', 'w') as f:
#     json.dump(dict_users, f)
with open('data'+bias+'.json', 'r') as f:
    dict_users = json.load(f)



epoch = 0
if args.model == 'cnn' and args.dataset == 'cifar':
    net_glob_init = CNNCifar(args=args).to(args.device)#CNNCifar_Compare().to(args.device)
elif args.model == 'cnn' and args.dataset == 'mnist':
    net_glob_init = CNNMnist_Compare().to(args.device)
elif args.model == 'mlp':
    len_in = 1
    for x in img_size:
        len_in *= x
    net_glob_init = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
else:
    exit('Error: unrecognized model')
net_glob_init.apply(weigth_init)
print(net_glob_init)
net_glob_init.train()
w_glob_init = net_glob_init.state_dict() 
idxs_users = [i for i in range(args.num_users)]

with open('device_index'+bias+'.json', 'r') as f:
    clusteded_dic = json.load(f)

env_id = "Federated-v0"
env = gym.make(env_id)

cluster_size = []
for i in range(10):
    j = '%d'%(i)
    cluster_size.append(len(clusteded_dic[j]))



preset_accuracy = torch.tensor([[55.0]])

n_episodes= 2000
iterations_per_episode = 50 
eps_start=1.0
eps_end = 0.01
eps_decay=0.99
verbose = False
all_rewards = [-10000000]
length_list = [1000000]
scores = [] # list containing score from each episode
#last_N_scores = deque(maxlen=20) # rollign window of last N scores
eps = eps_start
each_num = 4;

name1 = 'TrainingRecords_test_'+str(each_num*10)+bias+'_'
name2 = 'Device_list_cifar'+bias



max_score=-20-1
for ep_iter in range(1):
    score = 0
    policy_loss_total = 0
    w_list = []
    policy_reward = []
    length_temp = 0
    epoch = 0

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob_init = CNNCifar(args=args).to(args.device)#CNNCifar_Compare().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob_init = CNNMnist_Compare().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob_init = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob_init.apply(weigth_init)
    #torch.save(net_glob_init.state_dict(),'test_cifar'+bias+'.pkl')
    print(net_glob_init)
    net_glob_init.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('test_cifar.pkl', map_location='cpu')
    CUDA_DEVICES = 0, 1
    
    net_glob_init = torch.nn.DataParallel(net_glob_init, device_ids=CUDA_DEVICES)
    net_glob_init .load_state_dict(checkpoint,False)
    
    w_glob_init = net_glob_init.state_dict() 
    idxs_users = [i for i in range(args.num_users)]
    state_matrix_init,net_glob_init,acc_list_init,_,w_list_init= local_update(args,dataset_train,
                                                dataset_test,dict_users,idxs_users,epoch,net_glob_init,acc_list = None,preset = preset_accuracy)
    
    
    
    print('begin distance') 
    print('begin distance') 
    print('begin distance')         
    done = 0
    epoch = 1        
    for roll_num in range(5,20):
        policy_reward = []
        state_matrix = copy.deepcopy(state_matrix_init)
        net_glob = copy.deepcopy(net_glob_init)
        acc_list = copy.deepcopy(acc_list_init)
        w_list = copy.deepcopy(w_list_init)
        w_glob = net_glob.state_dict() 
        epoch = 1
        done = 0
        record_list = []
        print('roll out')        
        while not done: 

            state_list = feature_cal_norm1(w_list, w_glob)
            if each_num != 10:
                action_n = np.array([get_action(clusteded_dic,state_list,i,each_num) for i in range(10)])
                idxs_users = np.concatenate(action_n)
                idxs_users = idxs_users.tolist()
                if len(idxs_users) < each_num * 10:
                    need_num = each_num * 10 - len(idxs_users)
                    rest_devices = np.setdiff1d(np.arange(100), idxs_users)
                    sort_array = np.concatenate((rest_devices.reshape(-1,1),state_list[rest_devices].reshape(-1,1)),1)
                    sort_result = np.argsort(sort_array[:,1])[-need_num:]
                    idxs_users.extend(sort_array[:,0][sort_result])
                record_list.append(idxs_users)
            else:
                idxs_users = np.arange(100).tolist()

            next_state_matrix,net_glob,acc_list,w_list, reward_n,done_n= local_update(args,dataset_train,dataset_test,
                                       dict_users,idxs_users,epoch,net_glob,acc_list = acc_list, w_list = w_list, name =name1+'dis_cifar_'+str(roll_num)+'.txt',preset = preset_accuracy)            


            w_glob = net_glob.state_dict()
            

            if epoch >= 80:
                done_n = [1] * 80
                done = 1

            epoch += 1
            score += reward_n
            done = done_n[0]
        if each_num != 10:
            record_list = np.concatenate(record_list).reshape(-1,each_num*10)
            np.savetxt(name2+'dis_cifar_'+str(each_num*10)+'_'+str(roll_num)+'.txt',record_list)