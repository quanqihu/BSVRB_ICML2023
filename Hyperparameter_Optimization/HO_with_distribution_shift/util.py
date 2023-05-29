import numpy as np
import torch


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def zero_grad(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            p.grad.data.zero_()


def dic_to_array(dic, size):
    arr = np.zeros(size)
    for key in dic:
        arr[key - 1] = dic[key]
    return arr


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




#
# def proj_vec(v, Cfy):
#     temp_norm = np.linalg.norm(v)
#     if temp_norm > Cfy:
#         return (Cfy / temp_norm) * v
#     else:
#         return v
#
#
# def arr_prod(a):
#     prod = 1
#     for i in range(len(a)):
#         prod *= a[i]
#     return prod
#
#
# def proj_mat(V, Cgxy):
#     norm_V = np.linalg.norm(V)
#     if norm_V > Cgxy:
#         # s=s/norm_V
#         # return np.matmul(np.matmul(u,np.diag(s)),vh)
#         return (Cgxy / norm_V) * V
#     else:
#         return V
#
#
# def proj_lamb(H, lamb):
#     u, s, vh = np.linalg.svd(H)
#     if np.amin(s) < lamb:
#         s = (lamb / np.amin(s)) * s
#         # return np.matmul(np.matmul(u,np.diag(s)),vh)
#         return (lamb / np.amin(s)) * H
#     else:
#         return H
