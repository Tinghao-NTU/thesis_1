import gym
import matplotlib.pyplot as plt
import io
import torch 
import torch.nn as nn
import torch.optim as optim
from scipy.special import lambertw

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
from utils.update_device_LSTM_prob import local_update,get_state,permute_device
from models.UpdateProb import LocalUpdate
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
def generate_shadow_fading(mean,sigma,size):
    sigma=pow(10,sigma/10)
    mean=pow(10,mean/10)
    m = np.log(pow(mean,2)/np.sqrt(pow(sigma,2)+pow(mean,2)))
    sigma= np.sqrt(np.log(pow(sigma,2)/pow(mean,2)+1))
    np.random.seed(0)
    Lognormal_fade=np.random.lognormal(m,sigma,size)
    return Lognormal_fade


def func_p(x):
    return sum(0.01 * np.array(gtk) * np.sqrt(rou / ((1-rou) * Tk + x))) - 1

def binary(func,convergence, left, right,index = None):
#     print('current acceptable error: ' + str(convergence) + '\n')
    error = convergence + 1  
    cur_root = left
    count = 1
    while error > convergence:
        if abs(func(left)) < convergence:
            return left
        elif abs(func(right)) < convergence:
            return right
        else:
#             print(str(count) + ' root = ' +str(cur_root))
            middle = (left + right) / 2
            if (func(left) * func(middle)) < 0:
                right = middle
            else:
                left = middle
            cur_root = left
        error = abs(func(cur_root))
        count += 1
        if count > 100:
            #print('There is no root!')
            return cur_root
    return cur_root
all_clients = 100;
n = all_clients
np.random.seed(0)
random_num = np.random.random(all_clients)
dist = (50+200*random_num)*0.001 #250*1.414*np.random.random(50);
dist1 = 50*0.001
dist0 = 10*0.001

# dist = np.loadtxt('distance.txt')



plm = 128.1 + 37.6*np.log10(dist)

rd_number = generate_shadow_fading(0,8,n)

PathLoss_User_BS=plm+rd_number

g = pow(10,(-PathLoss_User_BS/10))

N0 =-174
N0 =pow(10,(N0/10))/1e3


i = np.array([i for i in range(0,all_clients)])
C = (9e3/100)*i + 1e3
np.random.seed(0)
np.random.shuffle(C)


B = 20*1e6
num_clients = 100
n = num_clients
f_max = 2*1e9
f_list = np.ones(num_clients) * 1e9
f_min = 0.2*1e9
p_max = 0.19952
p_min = 0.01
p = np.ones(num_clients) *0.19952





delta = 0.1
xi = 0.1 
epsilon = 0.001 
s = 4894444



alpha = 2e-28 

D = 500    
k = 1e-28   

t_cmp = 10.5 * C * D / f_list



E =  0.12251*np.ones(num_clients)

T0 = 100
b_init= np.ones(num_clients) * (B/num_clients)
b = b_init

eta = 0.5
gamma = 1
l = 1
a = (np.log2(1/epsilon)*2*pow(l,2))/(gamma*gamma*xi)
v = 2/((2-l*delta)*delta*gamma)

Ik = v*np.log2(1/eta)
I0 = a/(1-eta)

A = g * p / N0#s / np.log2(1+g*p/(N0*b)) 
F = v * C * D *np.log2(1/eta) 
G = v  * k * C * D * np.log2(1/eta)
H = (s * p)#/np.log2(1+g*p/(N0*b)) 

fenmu = B * np.log2(1+A/B)
R = np.log2(1+A/B)
s = 4894444#28800#
Tk =  4894444/fenmu



i = np.array([i for i in range(0,all_clients)])
C = (9e3/100)*i + 1e3
np.random.seed(0)
np.random.shuffle(C)

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
        
    

bias = '_05'

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
each_num = 1;

name1 = 'TrainingRecords_test_'+str(each_num*10)+bias+'_'
name2 = 'Device_list_prob'+bias

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
#     torch.save(net_glob_init.state_dict(),'test_cifar.pkl')
    print(net_glob_init)
    net_glob_init.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('test_cifar_08.pkl', map_location='cpu')
    #CUDA_DEVICES = 0, 1, 2
    
    #net_glob_init = torch.nn.DataParallel(net_glob_init, device_ids=CUDA_DEVICES)
    net_glob_init.load_state_dict(checkpoint,False)
    
    w_glob_init = net_glob_init.state_dict() 
    idxs_users = [i for i in range(args.num_users)]
    state_matrix_init,net_glob_init,acc_list_init,_,w_list_init,gtk= local_update(args,dataset_train,
                    dataset_test,dict_users,idxs_users,epoch,net_glob_init,acc_list = None,preset = preset_accuracy, prob = 1)
    
    
    
    print('begin distance') 
    print('begin distance') 
    print('begin distance')         
    done = 0
    epoch = 1        
    for roll_num in range(5,10):
        policy_reward = []
        T = 0
        E = 0
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
            

            rou = 0.9
            lamuda = binary(func_p,1e-2, 1e-6, 1,index = None)
            pk = 0.01 * np.array(gtk) * np.sqrt(rou / ((1-rou) * Tk + lamuda))            
            idxs_users = torch.multinomial(torch.from_numpy(pk), each_num * 10).tolist()
            b_list = B/(R[idxs_users] * sum(1/R[idxs_users]))
            fenmu = b_list * np.log2(1+A[idxs_users]/b_list)
            
            
            T += max((s /fenmu + F[idxs_users]/ f_list[idxs_users]).reshape(-1)) 
            E += sum((G[idxs_users] *f_list[idxs_users]*f_list[idxs_users]  + H[idxs_users]  / fenmu).tolist())

            
            state_list = feature_cal_norm1(w_list, w_glob)
            
            

            next_state_matrix,net_glob,acc_list,w_list, reward_n,done_n,gtk= local_update(args,dataset_train,dataset_test,
                        dict_users,idxs_users,epoch,net_glob,acc_list = acc_list, w_list = w_list, name =name1+'prob_cifar_'+str(roll_num)+'.txt',preset = preset_accuracy, gtk = gtk, prob = 1)            
            print(T)
            print(E)

            w_glob = net_glob.state_dict()
            

            if epoch >= 200:
                done_n = [1] * 80
                done = 1

            epoch += 1
            score += reward_n
            done = done_n[0]
        print('T is ' + str(T))
        print('E is ' + str(E))
        
        file_handle=open('T_prob.txt',mode='a')
        file_handle.write(str(T))
        file_handle.write('\n')
        file_handle.close()
        file_handle=open('E_prob.txt',mode='a')
        file_handle.write(str(E))
        file_handle.write('\n')
        file_handle.close()        
        
