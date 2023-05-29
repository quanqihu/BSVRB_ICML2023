import numpy as np
from libsvm.svmutil import *

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

def Datasets(dataset):
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

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, tr_num, val_num


