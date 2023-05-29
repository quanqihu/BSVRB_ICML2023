'''
In this project we consider BSVRB and RSVRB for solving hyperparameter optimization with linear model.

'''
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
import argparse
import os
import pickle

from util import set_all_seeds
from datasets import Datasets

parser = argparse.ArgumentParser(description='BSVRB_linear_ht')

### Parameter Setting
parser.add_argument('--SEED', default=123, type=int)
parser.add_argument('--total_epoch', default=10, type=int)

parser.add_argument('--method', default='BSVRBv1', type=str, help='BSVRBv1, BSVRBv2')
parser.add_argument('--dataset', default='a8a', type=str, help='a8a, w8a')
parser.add_argument('--task_num', default=100, type=int)
parser.add_argument('--data_batch_size', default=32, type=int)
parser.add_argument('--task_batch_size', default=10, type=int)
parser.add_argument('--lamb', default=0.00001, type=float, help='regularization parameter')
parser.add_argument('--eta', default=0.001, type=float, help='learning rate for p')
parser.add_argument('--alpha', default=0.9, type=float, help='MSVR parameter 1')
parser.add_argument('--gamma', default=1, type=float, help='MSVR parameter 2')
parser.add_argument('--tau', default=0.5, type=float, help='learning rate for w')
parser.add_argument('--beta', default=0.9, type=float, help='STORM parameter')
parser.add_argument('--tau_v', default=0, type=float,help='learning rate for v, default is to set to be the same as --tau')
parser.add_argument('--H_bound', default=10, type=float, help='H projection domain')
parser.add_argument('--v_bound', default=10, type=float, help='v projection domain')
parser.add_argument('--stag_decay', default=False, type=bool,
                    help='whether to do stage decay for learning rate, if true, decay eta & tau at 50% total_epoch')
parser.add_argument('--eva_per', default=100, type=int, help='print evaluation per # of iterations')
parser.add_argument('--results_path', default='.../', help='path to save log file')


def logLoss(w, x, y, sigma):
    loss = np.log(1 + np.exp((-y / sigma) * np.dot(w, np.append([1], x))))
    return loss

def log_func(z):
    return 1 / (1 + np.exp(-z))

def log_func_der(z):
    return np.exp(z) / (1 + np.exp(z)) ** 2

def f_evaluation(w, X_val, Y_val):
    loss = 0
    m = len(w)
    val_num = len(Y_val)
    for i in range(m):
        sigma = sigma_list[i]
        for j in range(len(Y_val)):
            loss += logLoss(w[i], X_val[j], Y_val[j], sigma)
    return loss / (m * val_num)

def g_evaluation_i(p, w, X_train, Y_train, lamb, i):
    loss = 0
    tr_num = len(Y_train)
    for j in range(tr_num):
        loss += log_func(p[j]) * logLoss(w, X_train[j], Y_train[j], sigma_list[i])
    reg = lamb / 2 * np.dot(w[i], w[i])
    return loss / tr_num + reg

def g_evaluation(p, w, X_train, Y_train, lamb, m):
    new_g_evaluation = 0
    for i in range(m):
        new_g_evaluation += g_evaluation_i(p, w, X_train, Y_train, lamb, i)
    return new_g_evaluation

def g_evaluation_i_noReg(p, w, X_train, Y_train, i):
    loss = 0
    tr_num = len(Y_train)
    for j in range(tr_num):
        loss += log_func(p[j]) * logLoss(w, X_train[j], Y_train[j], sigma_list[i])
    return loss / tr_num

def g_evaluation_noReg(p, w, X_train, Y_train, m):
    new_g_evaluation = 0
    for i in range(m):
        new_g_evaluation += g_evaluation_i_noReg(p, w, X_train, Y_train, i)
    return new_g_evaluation

def grad_y_f(w, X_val, Y_val, i, data_bat_ind):
    sigma = sigma_list[i]
    grad_y_f = np.zeros_like(w[0])
    for j in data_bat_ind:
        x = np.append([1], X_val[j])
        y = Y_val[j]
        wx = np.dot(w[i], x)
        term1 = 1 - 1 / (1 + np.exp(-(y / sigma) * wx))
        grad_y_f += - term1 * (y / sigma) * x
    return grad_y_f / len(data_bat_ind)

def grad_y_g(p, w, X_train, Y_train, lamb, i, data_bat_ind):
    sigma = sigma_list[i]
    grad_y_g = np.zeros_like(w[0])
    for j in data_bat_ind:
        x = np.append([1], X_train[j])
        y = Y_train[j]
        wx = np.dot(w[i], x)
        term1 = 1 - 1 / (1 + np.exp(-(y / sigma) * wx))
        grad_y_g += -log_func(p[j]) * term1 * (y / sigma) * x
    return grad_y_g / len(data_bat_ind) + lamb * w[i]

def hess_xy_g(p, w, X_train, Y_train, i, data_bat_ind):
    sigma = sigma_list[i]
    hess_xy_g = np.zeros((len(p), len(w[0])))
    for j in data_bat_ind:
        x = np.append([1], X_train[j])
        y = Y_train[j]
        wx = np.dot(w[i], x)
        term1 = 1 - 1 / (1 + np.exp(-(y / sigma) * wx))
        j_row = - log_func_der(p[j]) * term1 * (y / sigma) * x
        hess_xy_g[j] = j_row
    return hess_xy_g / len(data_bat_ind)

def hess_yy_g(p, w, X_train, Y_train, lamb, i, data_bat_ind):
    sigma = sigma_list[i]
    hess_yy_g = np.zeros((len(w[0]), len(w[0])))
    for j in data_bat_ind:
        x = np.append([1], X_train[j])
        y = Y_train[j]
        wx = np.dot(w[i], x)
        term_exp = np.exp(-(y / sigma) * wx)
        term1 = log_func(p[j]) * (y / sigma) ** 2 * (-term_exp / (1 + term_exp) ** 2)
        hess_yy_g += term1 * np.matmul(np.transpose([x]), [x])
    return hess_yy_g / len(data_bat_ind) + lamb * np.identity(len(w[0]))

def grad_v_gamma(p, w, v, X_train, Y_train, X_val, Y_val, lamb, i, data_bat_ind_tr, data_bat_ind_val):
    return np.transpose(np.matmul(hess_yy_g(p, w, X_train, Y_train, lamb, i, data_bat_ind_tr), v[i]))[0] - grad_y_f(w, X_val, Y_val, i, data_bat_ind_val)

def proj_mat(H, bound):
    norm_H = np.linalg.norm(H)
    if norm_H < bound:
        return (bound / norm_H) * H
    else:
        return H


def main():
    global args
    args = parser.parse_args()
    print('Running: ', args.method, ' on dataset: ', args.dataset, flush=True)

    global sigma_list
    sigma_path = 'sig_list.p'
    with open(sigma_path, 'rb') as handle:
        sigma_list = pickle.load(handle)
        handle.close()

    if args.method == 'BSVRBv1' or args.method == 'BSVRBv2':
        output_dict = BSVRB()
    elif args.method == 'RSVRB':
        output_dict = RSVRB()

    ### save log file
    file_name = 'linearHT_{0}_{1}.p'.format(args.dataset, args.method)
    results_path = args.results_path
    new_file_path = os.path.join(results_path, file_name)
    with open(new_file_path, 'wb') as handle:
        pickle.dump(output_dict, handle)
        handle.close()


def BSVRB():
    m = args.task_num
    lamb = args.lamb
    data_batch_size = args.data_batch_size
    task_batch_size = args.task_batch_size
    eta = args.eta
    alpha = args.alpha
    gamma = args.gamma
    beta = args.beta
    tau = args.tau
    if args.tau_v == 0:
        tau_v = tau
    else:
        tau_v = args.tau_v
    v_bound = args.v_bound

    print('number of tasks = ', m, ', data_bat_size = ', data_batch_size, ', task_bat_size = ', task_batch_size,
          flush=True)
    print('eta = ', eta, ', tau = ', tau, ', alpha = ', alpha, ', beta = ', beta, ', gamma = ', gamma, ', lamb = ',
          lamb, ', tau_v = ', tau_v, flush=True)

    ### Reproductivity
    set_all_seeds(args.SEED)

    ### Data processing
    X_train, Y_train, X_val, Y_val, X_test, Y_test, tr_num, val_num = Datasets(args.dataset)
    tr_data_ind = np.linspace(0, tr_num - 1, tr_num).astype(int)  ### for training data sampling
    val_data_ind = np.linspace(0, val_num - 1, val_num).astype(int)  ### for validation data sampling
    task_ind = np.linspace(0, m - 1, m).astype(int)  # for task sampling
    iter_per_epoch_tr = tr_num // (data_batch_size * task_batch_size)
    iter_per_epoch_val = val_num // (data_batch_size * task_batch_size)
    iter_per_epoch_task = m // task_batch_size

    ### linear model
    feature_num = len(X_train[0])  ### number of features
    w_dim = feature_num + 1

    ### Intialize variables
    p_0 = np.random.rand(tr_num)
    w_0 = np.random.rand(m, w_dim)  ### linear model: w^Tx+w0
    v_0 = np.random.rand(m, w_dim)
    H_0 = np.random.rand(m, w_dim, w_dim)

    p_1 = np.random.rand(tr_num)
    w_1 = np.random.rand(m, w_dim)  ### linear model: w^Tx+w0
    v_1 = np.random.rand(m, w_dim)
    s_1 = np.random.rand(m, w_dim)
    u_1 = np.random.rand(m, w_dim)
    H_1 = np.random.rand(m, w_dim, w_dim)
    z_1 = np.random.rand(tr_num)

    f_eval_list = np.array([f_evaluation(w_1, X_val, Y_val)])
    g_eval_list = np.array([g_evaluation(p_1, w_1, X_train, Y_train, lamb, m)])
    g_eval_noReg_list = np.array([g_evaluation_noReg(p_1, w_1, X_train, Y_train, m)])
    test_eval_list = np.array([f_evaluation(w_1, X_test, Y_test)])
    time_list = np.array([0])
    epoch_list = np.array([0])
    ite_list = np.array([0])

    epoch = 0  ### count the epoch number in terms of training dataset
    iter_num = 0
    timer_start = dt.now()
    decay_flag = 1
    decay_epoch = args.total_epoch // 2
    delay_count = np.zeros(m)

    while epoch <= args.total_epoch:
        ### stage wise decay eta and tau
        if args.stag_decay and (decay_flag == 1):
            if epoch == decay_epoch:
                eta = eta * 0.1
                tau = tau * 0.1
                tau_v = tau_v * 0.1
                decay_flag = 0

        ### sample data and task batch
        ### data shuffling after corresponding epoch ends
        tr_bat_ind = iter_num % iter_per_epoch_tr
        if tr_bat_ind == 0:
            epoch += 1
            np.random.shuffle(tr_data_ind)
        tr_bat_all = tr_data_ind[tr_bat_ind * data_batch_size * task_batch_size: (tr_bat_ind + 1) * data_batch_size * task_batch_size]

        val_bat_ind = iter_num % iter_per_epoch_val
        if val_bat_ind == 0:
            np.random.shuffle(val_data_ind)
        val_bat_all = val_data_ind[val_bat_ind * data_batch_size * task_batch_size: (val_bat_ind + 1) * data_batch_size * task_batch_size]

        task_bat_ind = iter_num % iter_per_epoch_task
        if task_bat_ind == 0:
            np.random.shuffle(task_ind)
        task_bat = task_ind[task_bat_ind * task_batch_size: (task_bat_ind + 1) * task_batch_size]

        ### Updates
        H_new = H_1.copy()
        w_new = w_1.copy()
        v_new = v_1.copy()
        G_0 = np.zeros_like(p_0)
        G_1 = np.zeros_like(p_0)
        for t in range(task_batch_size):
            task = task_bat[t]
            tr_bat = tr_bat_all[t * data_batch_size: (t + 1) * data_batch_size]
            val_bat = val_bat_all[t * data_batch_size: (t + 1) * data_batch_size]

            ### update s
            grad_gy_new = grad_y_g(p_1, w_1, X_train, Y_train, lamb, task, tr_bat)
            grad_gy_old = grad_y_g(p_0, w_0, X_train, Y_train, lamb, task, tr_bat)
            s_1[task] = (1 - alpha) * s_1[task] + (alpha + gamma) * grad_gy_new - gamma * grad_gy_old

            ### For BSVRBv2, update u
            if args.method == 'BSVRBv2':
                grad_v_gamma_new = grad_v_gamma(p_1, w_1, v_1, X_train, Y_train, X_val, Y_val, lamb, task, tr_bat,
                                                val_bat)
                grad_v_gamma_old = grad_v_gamma(p_0, w_0, v_0, X_train, Y_train, X_val, Y_val, lamb, task, tr_bat,
                                                val_bat)
                u_1[task] = (1 - alpha) * u_1[task] + (alpha + gamma) * grad_v_gamma_new - gamma * grad_v_gamma_old

            ### For BSVRBv1, update H
            if args.method == 'BSVRBv1':
                ### update Hessian
                grad_gyy_new = hess_yy_g(p_1, w_1, X_train, Y_train, lamb, task, tr_bat)
                grad_gyy_old = hess_yy_g(p_0, w_0, X_train, Y_train, lamb, task, tr_bat)
                H_new[task] = proj_mat(
                    (1 - alpha) * H_new[task] + (alpha + gamma) * grad_gyy_new - gamma * grad_gyy_old, args.H_bound)

            ### compute G
            hess_xy_g_new = hess_xy_g(p_1, w_1, X_train, Y_train, task, tr_bat)
            hess_xy_g_old = hess_xy_g(p_0, w_0, X_train, Y_train, task, tr_bat)
            if args.method == 'BSVRBv1':
                grad_y_f_new = grad_y_f(w_1, X_val, Y_val, task, val_bat)
                grad_y_f_old = grad_y_f(w_0, X_val, Y_val, task, val_bat)
                G_0 += -np.transpose(
                    np.matmul(np.matmul(hess_xy_g_old, np.linalg.inv(H_0[task])), np.transpose([grad_y_f_old])))[0]
                G_1 += -np.transpose(
                    np.matmul(np.matmul(hess_xy_g_new, np.linalg.inv(H_1[task])), np.transpose([grad_y_f_new])))[0]
            elif args.method == 'BSVRBv2':
                G_0 += - np.transpose(np.matmul(hess_xy_g_old, v_0[task]))[0]
                G_1 += - np.transpose(np.matmul(hess_xy_g_new, v_1[task]))[0]
            else:
                raise 'UnknownMethod'

        for task in range(m):
            if args.method == 'BSVRBv1':
                ### update w
                w_new[task] = w_new[task] - tau * s_1[task]
            elif args.method == 'BSVRBv2':
                if task in task_bat:
                    ### number of iteration that this task has missed
                    dc = delay_count[task] + 1
                    ### update w
                    w_new[task] = w_new[task] - tau * dc * s_1[task]
                    ### update v
                    temp_v_new = v_new[task] - tau_v * dc * u_1[task]
                    temp_v_new_norm = np.linalg.norm(temp_v_new)
                    v_new[task] = temp_v_new / temp_v_new_norm * v_bound if temp_v_new_norm > v_bound else temp_v_new
                    ### reset delay_count
                    delay_count[task] = 0
                else:
                    ### if task is unsampled, add 1 to delay_count
                    delay_count[task] += 1

        ### update z and p
        p_0 = p_1.copy()
        z_1 = (1 - beta) * (z_1 - G_0 / task_batch_size) + G_1 / task_batch_size
        p_1 = p_1 - eta * z_1

        ### update H_0, H_1, w_0, w_1, v_0, v_1
        H_0, H_1 = H_1.copy(), H_new.copy()
        w_0, w_1 = w_1.copy(), w_new.copy()
        if args.method == 'BSVRBv2':
            v_0, v_1 = v_1.copy(), v_new.copy()

        ### save to log
        iter_num += 1
        if iter_num % args.eva_per == 0:
            f_eval_list = np.append(f_eval_list, f_evaluation(w_1, X_val, Y_val))
            g_eval_list = np.append(g_eval_list, g_evaluation(p_1, w_1, X_train, Y_train, lamb, m))
            g_eval_noReg_list = np.append(g_eval_noReg_list, g_evaluation_noReg(p_1, w_1, X_train, Y_train, m))
            test_eval_list = np.append(test_eval_list, f_evaluation(w_1, X_test, Y_test))
            time_list = np.append(time_list, td.total_seconds(dt.now() - timer_start))
            epoch_list = np.append(epoch_list, epoch)
            ite_list = np.append(ite_list, iter_num)

            print("Epoch: {}, iter_num: {}, f_eval: {:4f}, g_eval: {:4f}, g_eval_noReg: {:4f}, test_eval: {:4f}".format(
                epoch, iter_num, f_eval_list[-1], g_eval_list[-1], g_eval_noReg_list[-1], test_eval_list[-1]),
                  flush=True)

            if np.isnan(f_eval_list[-1]):
                break

    output_dict = {'f_eval_list': f_eval_list,
                   'g_eval_list': g_eval_list,
                   'g_eval_noReg_list': g_eval_noReg_list,
                   'time_list': time_list,
                   'epoch_list': epoch_list,
                   'ite_list': ite_list,
                   'test_eval_list': test_eval_list
                   }
    print('Finished')
    return output_dict


def RSVRB():
    m = args.task_num
    lamb = args.lamb
    data_batch_size = args.data_batch_size
    task_batch_size = args.task_batch_size
    eta = args.eta
    beta = args.beta
    tau = args.tau

    print('number of tasks = ', m, ', data_bat_size = ', data_batch_size, ', task_bat_size = ', task_batch_size,
          flush=True)
    print('eta = ', eta, ', tau = ', tau, ', beta = ', beta, ', lamb = ', lamb, flush=True)

    ### Reproductivity
    set_all_seeds(args.SEED)

    ### Data processing
    X_train, Y_train, X_val, Y_val, X_test, Y_test, tr_num, val_num = Datasets(args.dataset)
    tr_data_ind = np.linspace(0, tr_num - 1, tr_num).astype(int)  # for training data sampling
    val_data_ind = np.linspace(0, val_num - 1, val_num).astype(int)  # for validation data sampling
    task_ind = np.linspace(0, m - 1, m).astype(int)  # for task sampling
    task_ind2 = np.linspace(0, m - 1, m).astype(int)  # for task sampling
    iter_per_epoch_tr = tr_num // (data_batch_size * task_batch_size)
    iter_per_epoch_val = val_num // (data_batch_size * task_batch_size)
    iter_per_epoch_task = m // task_batch_size

    ### linear model
    feature_num = len(X_train[0])  # number of features
    w_dim = feature_num + 1

    ### Intialize variables
    p_0 = np.random.rand(tr_num)  # x
    w_0 = np.random.rand(m, w_dim)  # y

    p_1 = np.random.rand(tr_num)
    w_1 = np.random.rand(m, w_dim)  # linear model: w^Tx+w0
    v_1 = np.random.rand(m, w_dim)
    V_1 = np.random.rand(m, tr_num, w_dim)
    H_1 = np.random.rand(m, w_dim, w_dim)
    s_1 = np.random.rand(m, w_dim)
    d_1 = np.random.rand(tr_num)

    f_eval_list = np.array([f_evaluation(w_1, X_val, Y_val)])
    g_eval_list = np.array([g_evaluation(p_1, w_1, X_train, Y_train, lamb, m)])
    g_eval_noReg_list = np.array([g_evaluation_noReg(p_1, w_1, X_train, Y_train, m)])
    test_eval_list = np.array([f_evaluation(w_1, X_test, Y_test)])
    time_list = np.array([0])
    epoch_list = np.array([0])
    ite_list = np.array([0])

    epoch = 0  ### count the epoch number in terms of training dataset
    iter_num = 0
    timer_start = dt.now()

    while epoch <= args.total_epoch:
        ### sample data and task batch
        ### data shuffling after corresponding epoch ends
        tr_bat_ind = iter_num % iter_per_epoch_tr
        if tr_bat_ind == 0:
            epoch += 1
            np.random.shuffle(tr_data_ind)
        tr_bat_all = tr_data_ind[tr_bat_ind * data_batch_size * task_batch_size: (tr_bat_ind + 1) * data_batch_size * task_batch_size]

        val_bat_ind = iter_num % iter_per_epoch_val
        if val_bat_ind == 0:
            np.random.shuffle(val_data_ind)
        val_bat_all = val_data_ind[val_bat_ind * data_batch_size * task_batch_size: (val_bat_ind + 1) * data_batch_size * task_batch_size]

        task_bat_ind = iter_num % iter_per_epoch_task
        if task_bat_ind == 0:
            np.random.shuffle(task_ind)
            np.random.shuffle(task_ind2)
        task_bat = task_ind[task_bat_ind * task_batch_size: (task_bat_ind + 1) * task_batch_size]
        task_bat2 = task_ind2[task_bat_ind * task_batch_size: (task_bat_ind + 1) * task_batch_size]

        ### Updates
        H_new = H_1.copy()
        w_new = w_1.copy()
        v_new = v_1.copy()
        V_new = V_1.copy()
        G_0 = np.zeros_like(p_0)
        G_1 = np.zeros_like(p_0)

        task_count = 0
        for task in range(m):
            if task in task_bat:
                tr_bat = tr_bat_all[task_count * data_batch_size: (task_count + 1) * data_batch_size]
                val_bat = val_bat_all[task_count * data_batch_size: (task_count + 1) * data_batch_size]
                task_count += 1

                ### update s
                grad_gy_new = grad_y_g(p_1, w_1, X_train, Y_train, lamb, task, tr_bat)
                grad_gy_old = grad_y_g(p_0, w_0, X_train, Y_train, lamb, task, tr_bat)
                s_1[task] = (1 - beta) * (s_1[task] - grad_gy_old * (m / task_batch_size)) + grad_gy_new * (
                            m / task_batch_size)

                ### update v
                grad_y_f_new = grad_y_f(w_1, X_val, Y_val, task, val_bat)
                grad_y_f_old = grad_y_f(w_0, X_val, Y_val, task, val_bat)
                v_new[task] = (1 - beta) * (v_new[task] - grad_y_f_old * (m / task_batch_size)) + grad_y_f_new * (
                            m / task_batch_size)

                ### update Hessian V
                hess_xy_g_new = hess_xy_g(p_1, w_1, X_train, Y_train, task, tr_bat)
                hess_xy_g_old = hess_xy_g(p_0, w_0, X_train, Y_train, task, tr_bat)
                V_new[task] = (1 - beta) * (V_new[task] - hess_xy_g_old * (m / task_batch_size)) + hess_xy_g_new * (
                            m / task_batch_size)

                ### update Hessian H
                grad_gyy_new = hess_yy_g(p_1, w_1, X_train, Y_train, lamb, task, tr_bat)
                grad_gyy_old = hess_yy_g(p_0, w_0, X_train, Y_train, lamb, task, tr_bat)
                H_new[task] = proj_mat(
                    (1 - beta) * (H_new[task] - grad_gyy_old * (m / task_batch_size)) + grad_gyy_new * (
                                m / task_batch_size), args.H_bound)

            else:
                s_1[task] = (1 - beta) * s_1[task]
                v_new[task] = (1 - beta) * v_new[task]
                V_new[task] = (1 - beta) * V_new[task]
                H_new[task] = (1 - beta) * H_new[task]

            if task in task_bat2:
                ### compute G
                G_0 += - np.transpose(np.matmul(np.matmul(V_1[task], np.linalg.inv(H_1[task])), np.transpose([v_1[task]])))[0]
                G_1 += - np.transpose(np.matmul(np.matmul(V_new[task], np.linalg.inv(H_new[task])), np.transpose([v_new[task]])))[0]

            w_new[task] = w_new[task] - tau * s_1[task]

        ### update z and p
        p_0 = p_1.copy()
        d_1 = (1 - beta) * (d_1 - G_0 / task_batch_size) + G_1 / task_batch_size
        p_1 = p_1 - eta * d_1

        ### update H_0, H_1, w_0, w_1, v_0, v_1
        H_1 = H_new.copy()
        w_0, w_1 = w_1.copy(), w_new.copy()
        v_1 = v_new.copy()
        V_1 = V_new.copy()

        ### save to log
        iter_num += 1
        if iter_num % args.eva_per == 0:
            f_eval_list = np.append(f_eval_list, f_evaluation(w_1, X_val, Y_val))
            g_eval_list = np.append(g_eval_list, g_evaluation(p_1, w_1, X_train, Y_train, lamb, m))
            g_eval_noReg_list = np.append(g_eval_noReg_list, g_evaluation_noReg(p_1, w_1, X_train, Y_train, m))
            test_eval_list = np.append(test_eval_list, f_evaluation(w_1, X_test, Y_test))
            time_list = np.append(time_list, td.total_seconds(dt.now() - timer_start))
            epoch_list = np.append(epoch_list, epoch)
            ite_list = np.append(ite_list, iter_num)

            print("Epoch: {}, iter_num: {}, f_eval: {:4f}, g_eval: {:4f}, g_eval_noReg: {:4f}, test_eval: {:4f}".format(
                epoch, iter_num, f_eval_list[-1], g_eval_list[-1], g_eval_noReg_list[-1], test_eval_list[-1]),
                  flush=True)

            if np.isnan(f_eval_list[-1]):
                break

    output_dict = {'f_eval_list': f_eval_list,
                   'g_eval_list': g_eval_list,
                   'g_eval_noReg_list': g_eval_noReg_list,
                   'time_list': time_list,
                   'epoch_list': epoch_list,
                   'ite_list': ite_list,
                   'test_eval_list': test_eval_list
                   }
    print('Finished')
    return output_dict


if __name__ == '__main__':
    main()
