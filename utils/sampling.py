#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torchvision import datasets, transforms
def data_generate(class_index,used_index,num,label):
    if len(class_index[label]) < num:
        #print('not enough')
        class_index[label] = np.concatenate((class_index[label],used_index[label]), axis=0)
        used_index[label] = np.array([], dtype='int64')
    index = class_index[label][:num]
    class_index[label] = np.delete(class_index[label], np.arange(num))
    used_index[label] = np.concatenate((index,used_index[label]), axis=0)
    
    return index, class_index, used_index

def imagenet_noniid(train_ds, num_users,ratio,H=None):
    train_data = []
    train_label = []
    for i in range(len(train_ds)):
        train_data.append(train_ds[i][0].unsqueeze(0))
        train_label.append(train_ds[i][1])

    dataset_train = torch.cat(train_data,0)
    labels = torch.tensor(train_label).long()
    all_classes = np.unique(labels)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    used_index = {i: np.array([], dtype='int64') for i in range(10)}
    class_index = {i: np.array([], dtype='int64') for i in range(10)}
    for i in range(10):
        class_index[i] = np.where(labels == i)[0]
    num_imgs = len(dataset_train)/num_users
    pri_list = []
    pre_num = num_imgs * ratio
    other_num = num_imgs - pre_num
    for i in range(num_users):
        pri_class =np.random.randint(0,10)
        pri_list.append(pri_class)
        #print(pri_class)
        other_class = np.delete(all_classes, pri_class)
        if H is not None:
            other_class = np.random.choice(other_class, 1, replace=False).repeat(9)
        pri_index, class_index, used_index = data_generate(class_index,used_index,int(pre_num),pri_class)    
        other_index = []
        for j in other_class:
            other_index_temp, class_index, used_index = data_generate(class_index,used_index,int(np.ceil(other_num/len(other_class))),j)
            other_index.append(other_index_temp)

        other_index = np.concatenate(other_index, axis=0)
        temp = np.concatenate((pri_index,other_index), axis=0)
        np.random.shuffle(temp)
        dict_users[i] = temp.tolist()
    return dict_users, pri_list




def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

'''
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
'''

# def mnist_noniid(dataset, num_users,ratio):
#     num_imgs = len(dataset)/num_users
#     #ratio = 0.8
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(len(dataset))
#     labels = dataset.train_labels.numpy()
#     idxs_labels = np.vstack((idxs, labels))
#     all_classes = np.unique(idxs_labels[1,:])
#     # divide and assign
#     for i in range(num_users):
#         pri_class = np.random.randint(0,9)
#         other_class = np.delete(all_classes, pri_class)

#         pri_index = np.where(idxs_labels[1,:] == pri_class)[0]
#         other_index = np.where(idxs_labels[1,:] != pri_class)[0]

#         pri_num = int(num_imgs * ratio)
#         other_num = int(num_imgs * (1-ratio))

#         selected_pri_index = np.random.choice(pri_index, pri_num, replace=False)
#         selected_other_index = np.random.choice(other_index, other_num, replace=False)

#         dict_users[i] = np.concatenate((selected_pri_index,selected_other_index), axis=0)
#     return dict_users
def mnist_noniid(dataset, num_users,ratio,H=None):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    dataset = dataset_train
    #num_users = 100
    labels = dataset.train_labels.numpy()
    all_classes = np.unique(labels)

    #ratio = 0.8
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    used_index = {i: np.array([], dtype='int64') for i in range(10)}
    class_index = {i: np.array([], dtype='int64') for i in range(10)}
    for i in range(10):
        class_index[i] = np.where(labels == i)[0]
    num_imgs = len(dataset_train)/num_users
    pri_list = []
    pre_num = num_imgs * ratio
    other_num = num_imgs - pre_num
    for i in range(num_users):
        pri_class = np.random.randint(0,10)
        pri_list.append(pri_class)
        #print(pri_class)
        other_class = np.delete(all_classes, pri_class)
        if H is not None:
            other_class = np.random.choice(other_class, 1, replace=False).repeat(9)
        pri_index, class_index, used_index = data_generate(class_index,used_index,int(pre_num),pri_class)    
        other_index = []
        for j in other_class:
            other_index_temp, class_index, used_index = data_generate(class_index,used_index,int(np.ceil(other_num/len(other_class))),j)
            other_index.append(other_index_temp)

        other_index = np.concatenate(other_index, axis=0)
        temp = np.concatenate((pri_index,other_index), axis=0)
        np.random.shuffle(temp)
        dict_users[i] = temp.tolist()
    return dict_users, pri_list


def cifar_noniid(dataset, num_users,ratio,H=None):
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    dataset = dataset_train
    #num_users = 100
    labels = np.array(dataset.targets)
    all_classes = np.unique(labels)

    #ratio = 0.8
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    used_index = {i: np.array([], dtype='int64') for i in range(10)}
    class_index = {i: np.array([], dtype='int64') for i in range(10)}
    for i in range(10):
        class_index[i] = np.where(labels == i)[0]
    num_imgs = len(dataset_train)/num_users
    pri_list = []
    pre_num = num_imgs * ratio
    other_num = num_imgs - pre_num
    for i in range(num_users):
        pri_class = np.random.randint(0,10)
        pri_list.append(pri_class)
        #print(pri_class)
        other_class = np.delete(all_classes, pri_class)
        if H is not None:
            other_class = np.random.choice(other_class, 1, replace=False).repeat(9)
        pri_index, class_index, used_index = data_generate(class_index,used_index,int(pre_num),pri_class)    
        other_index = []
        for j in other_class:
            other_index_temp, class_index, used_index = data_generate(class_index,used_index,int(np.ceil(other_num/len(other_class))),j)
            other_index.append(other_index_temp)

        other_index = np.concatenate(other_index, axis=0)
        temp = np.concatenate((pri_index,other_index), axis=0)
        np.random.shuffle(temp)
        dict_users[i] = temp.tolist()
    return dict_users,pri_list

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def fashion_noniid(dataset, num_users,ratio,H=None):
    dataset_train = datasets.FashionMNIST(
        '../data/fashion', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))
    dataset_test = datasets.FashionMNIST(
            '../data/fashion', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
    dataset = dataset_train
    #num_users = 100
    pri_list = []
    labels = dataset.train_labels.numpy()
    all_classes = np.unique(labels)

    #ratio = 0.8
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    used_index = {i: np.array([], dtype='int64') for i in range(10)}
    class_index = {i: np.array([], dtype='int64') for i in range(10)}
    for i in range(10):
        class_index[i] = np.where(labels == i)[0]
    num_imgs = len(dataset_train)/num_users

    pre_num = num_imgs * ratio
    other_num = num_imgs - pre_num
    for i in range(num_users):
        pri_class = np.random.randint(0,10)
        pri_list.append(pri_class)
        #print(pri_class)
        other_class = np.delete(all_classes, pri_class)
        if H is not None:
            other_class = np.random.choice(other_class, 1, replace=False).repeat(9)
        pri_index, class_index, used_index = data_generate(class_index,used_index,int(pre_num),pri_class)    
        other_index = []
        for j in other_class:
            other_index_temp, class_index, used_index = data_generate(class_index,used_index,int(np.ceil(other_num/len(other_class))),j)
            other_index.append(other_index_temp)

        other_index = np.concatenate(other_index, axis=0)
        dict_users[i] = np.concatenate((pri_index,other_index), axis=0).tolist()
    return dict_users,pri_list




if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
