#coding=utf-8
# @Time : 20-12-24下午3:55 
# @Author : Honglian WANG

import os
import sys
import numpy as np
import math
from config import config

def mini_batch(X,L,Y,M):
    '''
    produce minibatch
    :param X: data [num_data, num_len, fea_dim], torch tensor
    :param L: array [real_len]
    :param Y: label [num_data, fea_dim] array
    :param M: mask tensor(num_data, num_len, 1) torch tensor
    :return: list of tuple, tuple = (x,mask, label)
    '''
    seed = 62
    np.random.seed(seed)
    m = X.shape[0]  # num of data
    mini_batches = []

    # permutation
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:]
    shuffled_M = M[permutation,:,:]
    shuffled_L = L[permutation]
    shuffled_Y = Y[permutation]


    # split dataset
    num_complete_minibatches = math.floor(m / config.mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * config.mini_batch_size:(k + 1) * config.mini_batch_size, :, :]
        mini_batch_M = shuffled_M[k * config.mini_batch_size:(k + 1) * config.mini_batch_size, :, :]
        mini_batch_S = shuffled_L[k * config.mini_batch_size:(k + 1) * config.mini_batch_size]
        mini_batch_Y = shuffled_Y[k * config.mini_batch_size:(k + 1) * config.mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_S, mini_batch_Y, mini_batch_M)
        mini_batches.append(mini_batch)


    if m % config.mini_batch_size != 0:
        # if any last after splition
        mini_batch_X = shuffled_X[config.mini_batch_size * num_complete_minibatches:, :, :]
        mini_batch_M = shuffled_M[config.mini_batch_size * num_complete_minibatches:, :, :]
        mini_batch_S = shuffled_L[config.mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[config.mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_S, mini_batch_Y, mini_batch_M)
        mini_batches.append(mini_batch)

    return mini_batches





def train_test_split(X,L,Y,M):
    '''
    produce minibatch
    :param X: data [num_data, num_len, efa_dim]
    :param M: mask matrix
    :param Y: label [1,num_data]
    :return: list of tuple, tuple = (x,mask, label)
    '''
    seed = 62
    np.random.seed(seed)
    m = X.shape[0]  # num of data

    # permutation
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:]
    shuffled_M = M[permutation,:,:]
    shuffled_L = L[permutation]
    shuffled_Y = Y[permutation]


    # split dataset
    num_complete_train = math.floor(m * config.train)
    train_X = shuffled_X[:num_complete_train, :, :]
    train_M = shuffled_M[:num_complete_train, :, :]
    train_L = shuffled_L[:num_complete_train]
    train_Y = shuffled_Y[:num_complete_train]

    test_X = shuffled_X[num_complete_train:, :, :]
    test_M = shuffled_M[num_complete_train:, :, :]
    test_L = shuffled_L[num_complete_train:]
    test_Y = shuffled_Y[num_complete_train:]


    return [train_X,train_L,train_Y,train_M], [test_X, test_L, test_Y,test_M]

