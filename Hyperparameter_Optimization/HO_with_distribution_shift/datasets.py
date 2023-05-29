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

def Datasets(dataset, add_noise=False, noise_perc=0.1, imb=False, imb_rate=0.3):
    if imb:
        if (dataset == 'a8a') and (imb_rate==0.3):
            if not add_noise:
                file_path = '/datasets/a8a_noise_0_imb_3_n.p'
            elif noise_perc == 0.1:
                file_path = '/datasets/a8a_noise_1_imb_3_n.p'
            elif noise_perc == 0.2:
                file_path = '/datasets/a8a_noise_2_imb_3_n.p'
            elif noise_perc == 0.3:
                file_path = '/datasets/a8a_noise_3_imb_3_n.p'
            elif noise_perc == 0.4:
                file_path = '/datasets/a8a_noise_4_imb_3_n.p'
        with open(file_path, 'rb') as handle:
            dict = pickle.load(handle)
            handle.close()

        X_train = dict['X_train']
        Y_train = dict['Y_train']
        X_val = dict['X_val']
        Y_val = dict['Y_val']
        X_test = dict['X_test']
        Y_test = dict['Y_test']

    else:
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

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, len(Y_train), len(Y_val)

