import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import copy
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
scaler_2 = StandardScaler() 
def feature_cal(wlist, wglobal):
    state_list = []
    for i in range(100):
        para_list = []
        for (_, parameters1),(_, parameters2) in zip(wlist[i].items(),wglobal.items()):
            para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 2,dim = 0).cpu().numpy().item())
        state_list.append(torch.tensor(para_list).view(1,-1))
#     para_list = torch.cat(para_list)
    return torch.cat(state_list,0)
def get_state(clusteded_dic,state,i):
    if len(np.shape(state)) <3:
        state = np.expand_dims(state,0)
    device_idx = np.array(clusteded_dic[str(i)])
    device_num = len(np.array(clusteded_dic[str(i)]))
    return state[:,device_idx,:]

def permute_device(action, clusteded_dic, cluster_index):
    cluster_index = '%d'%(cluster_index)
    real_index = action
    device_list = clusteded_dic[cluster_index]
    selected_device = device_list[real_index]
    index_list = np.delete(device_list,real_index)
    device_list = np.append(index_list,selected_device).tolist()
    clusteded_dic[cluster_index] = device_list
    return clusteded_dic
def cluster_devices(wlist):    
    dist_mx = []
    class_index = {i: np.array([], dtype='int64') for i in range(10)}
    
    for i in range(100):
        temp = copy.deepcopy(wlist[i])
        para_list1 = temp['layer_input.weight']
        dist_mx.append(para_list1.view(1,-1))
    dist_mx = torch.cat(dist_mx,0)
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(dist_mx.cpu())
    for i in range(10):
        class_index[i] = np.where(kmeans.labels_ == i)[0].tolist()
    return class_index


def local_update(args,dataset_train,dataset_test,dict_users,idxs_users,epoch,net_glob,acc_list = None,cluster_acc = None, w_list = None, name = None,preset  = None, coeff = None):
    w_locals_temp = []
    loss_locals = []
    reward_n = []
    if preset is None:
        preset  = 99
    if coeff is None:
        coeff = 1.2

    if epoch ==0:
        acc_list = [-100000]
        w_list = []
    elif epoch == 1:
        cluster_acc = []
    count = 0
    for idx in idxs_users:
        net_temp = copy.deepcopy(net_glob)
        idx_str = '%d'%(idx)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx_str])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        #result = w['update_device_MLP.weight'].view(1,-1)
        w_locals_temp.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        
        if epoch > 1:
            net_temp.load_state_dict(copy.deepcopy(w))
            acc_train, loss_train = test_img(net_temp, dataset_test, args)
#             print(acc_train)
#             print(max(cluster_acc[count]))
            reward_n.append(np.clip(5*(acc_train.numpy().item()- max(cluster_acc[count])),-5,5)*min(1,epoch* 0.1))
#             reward_n.append(np.clip((acc_train.numpy().item()- max(cluster_acc[count])),-1,1)*min(1,epoch* 0.1))
            cluster_acc[count].append(acc_train.numpy().item())
        elif epoch == 1:
            net_temp.load_state_dict(copy.deepcopy(w))
            acc_train, loss_train = test_img(net_temp, dataset_test, args)
            cluster_acc.append([acc_train.numpy().item()])         
        count += 1
        if epoch == 0:
            w_list.append(w)

        else:
            w_list[idx] = copy.deepcopy(w)

    w_glob = FedAvg(w_locals_temp)
    state_matrix = feature_cal(w_list, w_glob)

    net_glob.load_state_dict(w_glob)

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
#         acc_test, loss_test = test_img(net_glob, dataset_test, args)     
    acc_train, loss_train = test_img(net_glob, dataset_test, args)
    acc_list.append(acc_train)
    print('Round {:3d}, Average loss {:.3f}, Training accuracy {:.3f}'.format(epoch, loss_avg,acc_train))
    if name is not None:
        file_handle=open(name,mode='a')
        file_handle.write("Round:%.03f; Average loss: %.03f; Training accuracy:%.03f \n"%(epoch,loss_avg,acc_train))
        file_handle.close()



    if acc_train < preset :
        done_n = [0] * 10
    else:
        done_n = [1] * 10
    if epoch == 0 :
        clusteded_dic = cluster_devices(w_list)
        return state_matrix,net_glob, acc_list,clusteded_dic,w_list
    else:
        return state_matrix,net_glob, acc_list,cluster_acc,w_list,reward_n,done_n

# def get_state(clusteded_dic,state,i):
#     i = '%d'%(i)
#     device_idx = np.array(clusteded_dic[i])
#     return torch.cat(((state[device_idx],state[-1].view(1,-1))),0).view(-1).cpu().detach().numpy()