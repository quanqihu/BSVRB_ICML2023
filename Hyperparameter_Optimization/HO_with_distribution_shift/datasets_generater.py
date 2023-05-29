import copy

import pandas as pd
import numpy as np
from libsvm.svmutil import *
import re
from array import array
import scipy
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import math
import pickle

def dic_to_matrix(dic):
    max_ind = 1
    for i in range(len(dic)):
        key_list = np.array(list(dic[i].keys()))
        if len(key_list) == 0:
            continue
        if max_ind < np.amax(key_list):
            max_ind = np.amax(key_list)
    M = np.zeros((len(dic), max_ind))
    for i in range(len(dic)):
        M[i] = dic_to_array(dic[i], max_ind)
    return M

def dic_to_array(dic, size):
    arr = np.zeros(size)
    for key in dic:
        arr[key - 1] = dic[key]
    return arr

def main():
    ### keep imb_rate of the positive data in training dataset
    if dataset == 'a8a':
        tr_num = 18156
        val_num = 4540
        Y_train, X_train = svm_read_problem('datasets/a8a_training.txt')
        Y_test, X_test = svm_read_problem('datasets/a8a_testing.txt')
    elif dataset == 'w8a':
        tr_num = 39799
        val_num = 9950
        Y_train, X_train = svm_read_problem('datasets/w8a_training.txt')
        Y_test, X_test = svm_read_problem('datasets/w8a_testing.txt')
    else:
        print('Unknown dataset!')

    X_train = dic_to_matrix(X_train)
    X_test = dic_to_matrix(X_test)

    X_train, Y_train = X_train[:tr_num], Y_train[:tr_num]
    X_val, Y_val = X_train[-val_num:], Y_train[-val_num:]

    if dataset=='a8a':
        zero_col = np.transpose([ np.zeros(len(X_test))])
        X_test = np.append(X_test, zero_col, axis=1)


    ### Count positive samples in training data
    pos_count = np.sum(Y_train==(np.ones_like(Y_train)))
    print('Original pos/neg rate: ', pos_count/len(Y_train))
    pos_ind_set = []
    for i in range(len(Y_train)):
        if Y_train[i]==1:
            pos_ind_set.append(i)
    imb_rate = 0.3
    pos_remove_ind = random.sample(pos_ind_set, int(math.floor(pos_count * (1-imb_rate))))
    X_train = np.delete(X_train, np.array(pos_remove_ind), axis=0)
    Y_train = np.delete(Y_train, np.array(pos_remove_ind))



    m = len(Y_train)
    label_ind = np.ndarray.tolist(np.linspace(0, m - 1, m).astype(int))


    noise_perc = 0
    new_dict = {'X_train': X_train,
                'Y_train': Y_train,
                'X_val': X_val,
                'Y_val': Y_val,
                'X_test': X_test,
                'Y_test': Y_test, }
    new_file_path = '/datasets/a8a_noise_0_imb_3_n.p'
    with open(new_file_path, 'wb') as handle:
        pickle.dump(new_dict, handle)
        handle.close()


    noise_perc = 0.1
    flip_ind = random.sample(label_ind, int(math.floor(m * noise_perc)))
    Y_train_copy = copy.deepcopy(Y_train)
    for i in flip_ind:
        if Y_train[i]==1:
            Y_train_copy[i]=-1
        else:
            Y_train_copy[i]=1
    new_dict = {'X_train': X_train,
                'Y_train': Y_train_copy,
                'X_val': X_val,
                'Y_val': Y_val,
                'X_test': X_test,
                'Y_test': Y_test, }
    new_file_path = '/datasets/a8a_noise_1_imb_3_n.p'
    with open(new_file_path, 'wb') as handle:
        pickle.dump(new_dict, handle)
        handle.close()

    noise_perc = 0.2
    flip_ind = random.sample(label_ind, int(math.floor(m * noise_perc)))
    Y_train_copy = copy.deepcopy(Y_train)
    for i in flip_ind:
            if Y_train[i] == 1:
                Y_train_copy[i] = -1
            else:
                Y_train_copy[i] = 1
    new_dict = {'X_train': X_train,
                'Y_train': Y_train_copy,
                'X_val': X_val,
                'Y_val': Y_val,
                'X_test': X_test,
                'Y_test': Y_test, }
    new_file_path = '/datasets/a8a_noise_2_imb_3_n.p'
    with open(new_file_path, 'wb') as handle:
        pickle.dump(new_dict, handle)
        handle.close()

    noise_perc = 0.3
    flip_ind = random.sample(label_ind, int(math.floor(m * noise_perc)))
    Y_train_copy = copy.deepcopy(Y_train)
    for i in flip_ind:
        if Y_train[i] == 1:
            Y_train_copy[i] = -1
        else:
            Y_train_copy[i] = 1
    new_dict = {'X_train': X_train,
                'Y_train': Y_train_copy,
                'X_val': X_val,
                'Y_val': Y_val,
                'X_test': X_test,
                'Y_test': Y_test, }
    new_file_path = '/datasets/a8a_noise_3_imb_3_n.p'
    with open(new_file_path, 'wb') as handle:
        pickle.dump(new_dict, handle)
        handle.close()

    noise_perc = 0.4
    flip_ind = random.sample(label_ind, int(math.floor(m * noise_perc)))
    Y_train_copy = copy.deepcopy(Y_train)
    for i in flip_ind:
        if Y_train[i] == 1:
            Y_train_copy[i] = -1
        else:
            Y_train_copy[i] = 1
    new_dict = {'X_train': X_train,
                'Y_train': Y_train_copy,
                'X_val': X_val,
                'Y_val': Y_val,
                'X_test': X_test,
                'Y_test': Y_test, }
    new_file_path = '/datasets/a8a_noise_4_imb_3_n.p'
    with open(new_file_path, 'wb') as handle:
        pickle.dump(new_dict, handle)
        handle.close()


    return X_train, Y_train, X_val, Y_val, X_test, Y_test, tr_num, val_num


main()
