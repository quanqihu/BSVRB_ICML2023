'''
In this project we consider hyperparameter optimization with linear model.

'''
import torch
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import date
import argparse
import os
import pickle

from util import set_all_seeds
from datasets import Datasets


parser = argparse.ArgumentParser(description='BSVRB_linear_ht')

### Parameter Setting
parser.add_argument('--SEED', default=123, type=int)
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--total_epoch', default=10, type=int)
parser.add_argument('--total_iter', default=1000, type=int)
parser.add_argument('--save_period', default=100, type=int)

parser.add_argument('--method', default='BSVRBv1', type=str, help='BSVRBv1, BSVRBv2')
parser.add_argument('--dataset', default='a8a', type=str, help='a1a, a8a, w8a')
parser.add_argument('--task_num', default=100, type=int)
parser.add_argument('--data_batch_size', default=32, type=int)
parser.add_argument('--task_batch_size', default=10, type=int)
parser.add_argument('--sigma_option', default='rand', type=str)
parser.add_argument('--lamb', default=1, type=float, help='regulirization parameter')
parser.add_argument('--eta', default=0.01, type=float, help='step size for p')
parser.add_argument('--alpha', default=0.01, type=float, help='step size for s, H')
parser.add_argument('--auto_gamma', default=1, type=int, help='whether use the gamma setting in theory')  ### Default=True, add --auto_gamma 0 to set False
parser.add_argument('--gamma', default=0.01, type=float, help='MSVR parameter')
parser.add_argument('--tau', default=0, type=float, help='step size for w')
parser.add_argument('--beta', default=0.01, type=float, help='step size for z')
parser.add_argument('--v_proj', default=False, type=bool, help='whether do projection in v updates')
parser.add_argument('--v_bound', default=0, type=float, help='v projection domain')
parser.add_argument('--v1_fullUpdate', default=False, type=bool, help='whether do full updates for BSVRBv1')
parser.add_argument('--H_proj', default=False, type=bool, help='whether do projection in H updates')
parser.add_argument('--H_bound', default=0, type=float, help='H projection domain')
parser.add_argument('--v2_partialUpdate', default=False, type=bool, help='whether do partial updates for BSVRBv2')
parser.add_argument('--tau_v', default=0, type=float, help='step size for v')
parser.add_argument('--stag_decay', default=False, type=bool, help='whether do stage decay for learning rate')
parser.add_argument('--eva_per', default=1000, type=int, help='evaluate per # of iterations')
parser.add_argument('--data_noise', default=False, type=bool, help='whether to add 10% noise to training data')
parser.add_argument('--noise_perc', default=0.1, type=float, help='noise percentage')
parser.add_argument('--imb', default=False, type=bool, help='whether to use imbalanced data')
parser.add_argument('--imb_rate', default=0.3, type=float, help='imbalanced rate')
parser.add_argument('--results_path', default='.../', type=str, help='results path')



def logLoss(w,x,y,sigma):
    loss = np.log(1 + np.exp((-y/sigma) * np.dot(w, np.append([1], x))))
    return loss

def log_func(z):
    return 1 / (1 + np.exp(-z))

def log_func_der(z):
    return np.exp(z) / (1 + np.exp(z)) ** 2

def model_evaluation(model, X, Y):
    loss = 0
    data_num = len(Y)
    for j in range(data_num):
        loss +=  logLoss(model, X[j], Y[j], 1)
    return loss/data_num

def f_evaluation(w, X_val, Y_val):
    loss = 0
    m=len(w)
    val_num = len(Y_val)
    for i in range(m):
        sigma = sigma_list[i]
        for j in range(len(Y_val)):
            loss += logLoss(w[i], X_val[j], Y_val[j], sigma)
    return loss/(m*val_num)

def g_evaluation_i(p, w, X_train, Y_train, lamb,i):
    loss = 0
    tr_num = len(Y_train)
    for j in range(tr_num):
        loss += log_func(p[j]) * logLoss(w, X_train[j], Y_train[j], sigma_list[i])
    reg = lamb/2 * np.dot(w[i], w[i])
    return loss/tr_num+reg

def g_evaluation(p, w, X_train, Y_train, lamb, m):
    new_g_evaluation = 0
    for i in range(m):
        new_g_evaluation += g_evaluation_i(p, w, X_train, Y_train, lamb, i)
    return new_g_evaluation

def g_evaluation_i_noReg(p, w, X_train, Y_train,i):
    loss = 0
    tr_num = len(Y_train)
    for j in range(tr_num):
        loss += log_func(p[j]) * logLoss(w, X_train[j], Y_train[j], sigma_list[i])
    return loss/tr_num

def g_evaluation_noReg(p, w, X_train, Y_train, m):
    ### val_data_num=len(X_val)
    new_g_evaluation = 0
    for i in range(m):
        new_g_evaluation += g_evaluation_i_noReg(p, w, X_train, Y_train, i)
    return new_g_evaluation

def accuracy_eval(model, X, Y):
    pred_label=np.array([])
    data_num = len(Y)
    for i in range(data_num):
        if np.dot(model, np.append([1], X[i])) >=0:
            pred_label = np.append(pred_label, 1)
        else:
            pred_label = np.append(pred_label, -1)
    return np.sum(pred_label == Y)/data_num

def accuracy_eval_bestModel(w_1, X_val, Y_val, X_test, Y_test):
    model_accu_val_list = np.array([])
    model_accu_test_list = np.array([])
    for model_ind in range(len(w_1)):
        model_accu_val_list = np.append(model_accu_val_list, accuracy_eval(w_1[model_ind], X_val, Y_val))
        model_accu_test_list = np.append(model_accu_test_list, accuracy_eval(w_1[model_ind], X_test, Y_test))
    best_model_accu_ind = np.argmax(model_accu_val_list)

    model_loss_val_list = np.array([])
    model_loss_test_list = np.array([])
    for model_ind in range(len(w_1)):
        model_loss_val_list = np.append(model_loss_val_list, model_evaluation(w_1[model_ind], X_val, Y_val))
        model_loss_test_list = np.append(model_loss_test_list, model_evaluation(w_1[model_ind], X_test, Y_test))
    best_model_loss_ind = np.argmin(model_loss_val_list)
    return model_accu_val_list[best_model_accu_ind], model_accu_test_list[best_model_accu_ind], model_loss_val_list[best_model_loss_ind], model_loss_test_list[best_model_loss_ind]

def grad_y_f(w, X_val, Y_val, i, data_bat_ind):
    sigma = sigma_list[i]
    grad_y_f = np.zeros_like(w[0])
    for j in data_bat_ind:
        x = np.append([1],X_val[j])
        y = Y_val[j]
        wx = np.dot(w[i],x)
        term1 = 1-1/(1+np.exp(-(y/sigma)*wx))
        grad_y_f += - term1 * (y/sigma) * x
    return grad_y_f/len(data_bat_ind)

def grad_y_g(p, w, X_train, Y_train, lamb, i, data_bat_ind):
    sigma = sigma_list[i]
    grad_y_g = np.zeros_like(w[0])
    for j in data_bat_ind:
        x = np.append([1],X_train[j])
        y = Y_train[j]
        wx = np.dot(w[i],x)
        term1 = 1-1/(1+np.exp(-(y/sigma)*wx))
        grad_y_g += -log_func(p[j]) * term1 * (y/sigma) * x
    return grad_y_g/len(data_bat_ind) + lamb*w[i]

def hess_xy_g(p, w, X_train, Y_train, i, data_bat_ind):
    sigma = sigma_list[i]
    hess_xy_g=np.zeros((len(p),len(w[0])))
    for j in data_bat_ind:
        x = np.append([1],X_train[j])
        y = Y_train[j]
        wx = np.dot(w[i],x)
        term1 = 1-1/(1 + np.exp(-(y/sigma) * wx))
        j_row= - log_func_der(p[j]) * term1 * (y/sigma) * x
        hess_xy_g[j]=j_row
    return hess_xy_g/len(data_bat_ind)

def hess_yy_g(p, w, X_train, Y_train, lamb, i, data_bat_ind):
    sigma = sigma_list[i]
    hess_yy_g = np.zeros((len(w[0]), len(w[0])))
    for j in data_bat_ind:
        x = np.append([1],X_train[j])
        y = Y_train[j]
        wx = np.dot(w[i],x)
        term_exp = np.exp(-(y/sigma)*wx)
        term1 = log_func(p[j]) * (y/sigma)**2 * (-term_exp/(1+term_exp)**2)
        hess_yy_g += term1 * np.matmul(np.transpose([x]),[x])
    return hess_yy_g/len(data_bat_ind) + lamb*np.identity(len(w[0]))

def grad_v_gamma(p, w, v, X_train, Y_train, X_val, Y_val, lamb, i, data_bat_ind_tr, data_bat_ind_val):
    return np.transpose(np.matmul(hess_yy_g(p, w, X_train, Y_train, lamb, i, data_bat_ind_tr), v[i]))[0] - grad_y_f(w, X_val, Y_val, i, data_bat_ind_val)

def proj_mat(H, bound):
    norm_H=np.linalg.norm(H)
    if norm_H < bound:
        return (bound/norm_H)*H
    else:
        return H

def main():
    global args
    args = parser.parse_args()

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('Running: ', args.method, ' on dataset: ', args.dataset, flush=True)

    ### sigma option
    global sigma_list

    if args.task_num == 1:
        sigma_list = np.array([1])
    else:
        sigma_path = '/sig_list.p'
        with open(sigma_path, 'rb') as handle:
            sigma_list = pickle.load(handle)
            handle.close()

    if args.method == 'BSVRBv1' or args.method == 'BSVRBv2':
        output_dict = RSVRB()

    today = date.today()
    datestr = today.strftime("%b%d%Y")
    if args.v2_partialUpdate:
        v2pU=0
    else:
        v2pU = 1

    if args.stag_decay:
        dec=1
    else:
        dec=0

    if args.data_noise:
        noise_flag=1
    else:
        noise_flag=0

    if args.imb:
        imb_flag=1
    else:
        imb_flag=0


    file_name = 'linearHT_{0}_{1}.p'.format(args.dataset,args.method)
    results_path = args.results_path

    new_file_path = os.path.join(results_path, file_name)
    with open(new_file_path, 'wb') as handle:
        pickle.dump(output_dict, handle)
        handle.close()

def RSVRB():
    m = args.task_num
    lamb = args.lamb
    data_batch_size = args.data_batch_size
    task_batch_size = args.task_batch_size
    eta = args.eta
    alpha = args.alpha
    if (args.auto_gamma==1) and alpha==1:
        gamma = 0
    elif args.auto_gamma==1:
        gamma = (m-task_batch_size)/(task_batch_size*(1-alpha))
    else:
        gamma = args.gamma
    beta = args.beta
    tau = args.tau
    if args.tau_v==0:
        tau_v = tau
    else:
        tau_v = args.tau_v
    v_bound = args.v_bound

    eva_per = args.eva_per

    print('number of tasks = ', m, ', data_bat_size = ', data_batch_size, ', task_bat_size = ', task_batch_size, ', SEED = ', args.SEED, flush=True)
    print('auto_gamma', args.auto_gamma, flush=True)
    print('eta = ', eta, ', tau = ', tau, ', alpha = ', alpha, ', beta = ', beta, ', gamma = ', gamma, ', lamb = ', lamb, ', tau_v = ', tau_v, flush=True)
    print('data_noise = ', args.data_noise, 'noise_perc = ', args.noise_perc, flush=True)

    ### Reproductivity
    set_all_seeds(args.SEED)

    ### Data processing
    X_train, Y_train, X_val, Y_val, X_test, Y_test, tr_num, val_num = Datasets(args.dataset, add_noise=args.data_noise, noise_perc=args.noise_perc, imb=args.imb, imb_rate=args.imb_rate)
    tr_data_ind = np.linspace(0, tr_num - 1, tr_num).astype(int)   # for training data sampling
    val_data_ind = np.linspace(0, val_num - 1, val_num).astype(int)   # for validation data sampling
    task_ind = np.linspace(0, m - 1, m).astype(int)   # for task sampling
    iter_per_epoch_tr = tr_num// (data_batch_size*task_batch_size)
    iter_per_epoch_val = val_num // (data_batch_size*task_batch_size)
    iter_per_epoch_task = m // task_batch_size

    ### linear model
    feature_num = len(X_train[0]) # number of features
    w_dim = feature_num+1

    ### Intialize variables
    p_0 = np.random.rand(tr_num)
    w_0 = np.random.rand(m, w_dim)  # linear model: w^Tx+w0
    v_0 = np.random.rand(m, w_dim)
    H_0 = np.random.rand(m, w_dim, w_dim)

    p_1 = np.random.rand(tr_num)
    w_1 = np.random.rand(m, w_dim)  # linear model: w^Tx+w0
    v_1 = np.random.rand(m, w_dim)
    s_1 = np.random.rand(m, w_dim)
    u_1 = np.random.rand(m, w_dim)
    H_1 = np.random.rand(m, w_dim, w_dim)
    z_1 = np.random.rand(tr_num)

    f_eval_list = np.array([f_evaluation(w_1, X_val, Y_val)])
    g_eval_list = np.array([g_evaluation(p_1, w_1, X_train, Y_train, lamb, m)])
    g_eval_noReg_list = np.array([g_evaluation_noReg(p_1, w_1, X_train, Y_train, m)])
    test_eval_list = np.array([f_evaluation(w_1, X_test, Y_test)])


    accu_val, accu_test, loss_val, loss_test = accuracy_eval_bestModel(w_1, X_val, Y_val, X_test, Y_test)
    accuracy_val_list = np.array([accu_val])
    accuracy_test_list = np.array([accu_test])
    loss_val_list = np.array([loss_val])
    loss_test_list = np.array([loss_test])
    best_accuracy_val = accuracy_val_list[0]
    best_accuracy_test = accuracy_test_list[0]
    best_model = w_1
    best_loss_val = loss_val_list[0]
    best_loss_test = loss_test_list[0]


    time_list = np.array([0])
    epoch_list = np.array([0])
    ite_list = np.array([0])

    epoch = 0 # count the epoch number in terms of training dataset
    iter_num = 0
    timer_start = dt.now()
    evaluation_flag = 0
    decay_flag = 1
    decay_epoch = args.total_epoch//2

    print('Start training', flush=True)

    while epoch <= args.total_epoch:
        ### stage wise decay tau
        if args.stag_decay and (decay_flag==1):
            if epoch == decay_epoch:
                tau = tau*0.1
                decay_flag=0

        ### sample data and task batch
        ### data shuffling after corresponding epoch ends
        tr_bat_ind = iter_num % iter_per_epoch_tr
        if tr_bat_ind == 0:
            epoch += 1
            np.random.shuffle(tr_data_ind)
            evaluation_flag = 1
        tr_bat_all = tr_data_ind[tr_bat_ind*data_batch_size*task_batch_size : (tr_bat_ind+1)*data_batch_size*task_batch_size]

        val_bat_ind = iter_num % iter_per_epoch_val
        if val_bat_ind == 0:
            np.random.shuffle(val_data_ind)
        val_bat_all = val_data_ind[val_bat_ind * data_batch_size*task_batch_size : (val_bat_ind + 1) * data_batch_size*task_batch_size]

        task_bat_ind = iter_num % iter_per_epoch_task
        if task_bat_ind == 0:
            np.random.shuffle(task_ind)
        task_bat = task_ind[task_bat_ind * task_batch_size : (task_bat_ind+1) * task_batch_size]

        ### Updates
        H_new = H_1.copy()
        w_new = w_1.copy()
        v_new = v_1.copy()
        G_0 = np.zeros_like(p_0)
        G_1 = np.zeros_like(p_0)
        for t in range(task_batch_size):
            task = task_bat[t]
            tr_bat = tr_bat_all[t*data_batch_size : (t+1)*data_batch_size]
            val_bat = val_bat_all[t * data_batch_size: (t + 1) * data_batch_size]

            ### update s
            grad_gy_new = grad_y_g(p_1,w_1,X_train,Y_train,lamb,task,tr_bat)
            grad_gy_old = grad_y_g(p_0,w_0,X_train,Y_train,lamb,task,tr_bat)
            s_1[task] = (1-alpha)*s_1[task] + (alpha+gamma)*grad_gy_new - gamma*grad_gy_old

            ### For RSVRBv2, update u
            if args.method == 'RSVRBv2':
                grad_v_gamma_new = grad_v_gamma(p_1,w_1,v_1,X_train,Y_train,X_val,Y_val,lamb,task,tr_bat,val_bat)
                grad_v_gamma_old = grad_v_gamma(p_0,w_0,v_0,X_train,Y_train,X_val,Y_val,lamb,task,tr_bat,val_bat)
                u_1[task] = (1-alpha)*u_1[task] + (alpha+gamma)*grad_v_gamma_new - gamma*grad_v_gamma_old

            ### For RSVRBv1, update w and H
            if args.method == 'RSVRBv1':
                if not args.v1_fullUpdate:
                    ### update w
                    w_new[task] = w_new[task] - tau * s_1[task]
                ### update Hessian
                grad_gyy_new = hess_yy_g(p_1,w_1,X_train,Y_train,lamb,task,tr_bat)
                grad_gyy_old = hess_yy_g(p_0,w_0,X_train,Y_train,lamb,task,tr_bat)
                if args.H_proj:
                    H_new[task] = proj_mat( (1 - alpha) * H_new[task] + (alpha + gamma) * grad_gyy_new - gamma * grad_gyy_old, args.H_bound)
                else:
                    H_new[task] = (1 - alpha) * H_new[task] + (alpha + gamma) * grad_gyy_new - gamma * grad_gyy_old

            ### For RSVRBv2 partial, update w and v
            if (args.method == 'RSVRBv2') and args.v2_partialUpdate:
                w_new[task] = w_new[task] - tau * s_1[task]
                if args.v_proj:
                    temp_v_new = v_new[task] - tau_v * u_1[task]
                    temp_v_new_norm = np.linalg.norm(temp_v_new)
                    v_new[
                        task] = temp_v_new / temp_v_new_norm * v_bound if temp_v_new_norm > v_bound else temp_v_new
                else:
                    v_new[task] = v_new[task] - tau_v * u_1[task]

            ### compute G
            hess_xy_g_new = hess_xy_g(p_1, w_1, X_train, Y_train, task, tr_bat)
            hess_xy_g_old = hess_xy_g(p_0, w_0, X_train, Y_train, task, tr_bat)
            if args.method == 'RSVRBv1':
                grad_y_f_new = grad_y_f(w_1,X_val,Y_val,task,val_bat)
                grad_y_f_old = grad_y_f(w_0,X_val,Y_val,task,val_bat)
                G_0 += - np.transpose(np.matmul(np.matmul(hess_xy_g_old, np.linalg.inv(H_0[task])), np.transpose([grad_y_f_old])))[0]
                G_1 += - np.transpose(np.matmul(np.matmul(hess_xy_g_new, np.linalg.inv(H_1[task])), np.transpose([grad_y_f_new])))[0]
            elif args.method == 'RSVRBv2':
                G_0 += - np.transpose(np.matmul(hess_xy_g_old, v_0[task]))[0]
                G_1 += - np.transpose(np.matmul(hess_xy_g_new, v_1[task]))[0]
            else:
                raise 'UnknownMethod'

        for task in range(m):
            if (args.method == 'RSVRBv2') and (not args.v2_partialUpdate):
                w_new[task] = w_new[task] - tau * s_1[task]
                if args.v_proj:
                    temp_v_new = v_new[task] - tau_v * u_1[task]
                    temp_v_new_norm = np.linalg.norm(temp_v_new)
                    v_new[
                        task] = temp_v_new / temp_v_new_norm * v_bound if temp_v_new_norm > v_bound else temp_v_new
                else:
                    v_new[task] = v_new[task] - tau_v * u_1[task]
            if args.v1_fullUpdate:
                w_new[task] = w_new[task] - tau * s_1[task]


        ### update z and p
        p_0 = p_1.copy()
        z_1 = (1-beta) * (z_1-G_0/task_batch_size) + G_1/task_batch_size
        p_1 = p_1 - eta * z_1

        ### update H_0, H_1, w_0, w_1, v_0, v_1
        H_0, H_1 = H_1.copy(), H_new.copy()
        w_0, w_1 = w_1.copy(), w_new.copy()
        if args.method == 'BSVRBv2':
            v_0, v_1 = v_1.copy(), v_new.copy()

        # save to log
        iter_num += 1
        # if evaluation_flag == 1:
        if iter_num % eva_per == 0:
            # f_eval_list = np.append(f_eval_list, f_evaluation(w_1, X_val, Y_val))
            # g_eval_list = np.append(g_eval_list, g_evaluation(p_1, w_1, X_train, Y_train, lamb, m))
            # g_eval_noReg_list = np.append(g_eval_noReg_list, g_evaluation_noReg(p_1, w_1, X_train, Y_train, m))
            # test_eval_list = np.append(test_eval_list, f_evaluation(w_1,X_test,Y_test))
            time_list=np.append(time_list,td.total_seconds(dt.now()-timer_start))
            epoch_list=np.append(epoch_list,epoch)
            ite_list=np.append(ite_list,iter_num)

            accu_val, accu_test, loss_val, loss_test = accuracy_eval_bestModel(w_1, X_val, Y_val, X_test, Y_test)
            accuracy_val_list = np.append(accuracy_val_list, accu_val)
            accuracy_test_list = np.array([accu_test])
            loss_val_list = np.append(accuracy_test_list, loss_val)
            loss_test_list = np.append(loss_test_list, loss_test)
            if best_accuracy_val < accu_val:
                best_accuracy_val = accu_val
                best_accuracy_test = accu_test
                best_model = w_1
                best_loss_val = loss_val
                best_loss_test = loss_test

            print("Epoch: {}, iter_num: {}, best_loss_val: {:4f}, best_loss_test: {:4f}, accuracy_val: {:4f}, accuracy_test: {:4f}, best_accuracy_val: {:4f}, best_accuracy_test: {:4f}, time: {:4f}".format(epoch,iter_num, best_loss_val, best_loss_test, accu_val, accu_test, best_accuracy_val, best_accuracy_test, time_list[-1]), flush=True)

            if np.isnan(loss_val_list[-1]):
                break

    time_list = np.append(time_list, td.total_seconds(dt.now() - timer_start))
    epoch_list = np.append(epoch_list, epoch)
    ite_list = np.append(ite_list, iter_num)

    accu_val, accu_test, loss_val, loss_test = accuracy_eval_bestModel(w_1, X_val, Y_val, X_test, Y_test)
    accuracy_val_list = np.append(accuracy_val_list, accu_val)
    accuracy_test_list = np.array([accu_test])
    loss_val_list = np.append(accuracy_test_list, loss_val)
    loss_test_list = np.append(loss_test_list, loss_test)
    if best_accuracy_val < accu_val:
        best_accuracy_val = accu_val
        best_accuracy_test = accu_test
        best_model = w_1
        best_loss_val = loss_val
        best_loss_test = loss_test

    print("Epoch: {}, iter_num: {}, best_loss_val: {:4f}, best_loss_test: {:4f}, accuracy_val: {:4f}, accuracy_test: {:4f}, best_accuracy_val: {:4f}, best_accuracy_test: {:4f}, time: {:4f}".format(epoch,iter_num, best_loss_val, best_loss_test, accu_val, accu_test, best_accuracy_val, best_accuracy_test, time_list[-1]), flush=True)

    output_dict = {'f_eval_list': f_eval_list,
                   'g_eval_list': g_eval_list,
                   'g_eval_noReg_list': g_eval_noReg_list,
                   'time_list': time_list,
                   'epoch_list': epoch_list,
                   'ite_list': ite_list,
                   'best_model': best_model,
                   'accuracy_val_list': accuracy_val_list,
                   'accuracy_test_list': accuracy_test_list,
                   'loss_val_list': loss_val_list,
                   'loss_test_list': loss_test_list
                   }
    print('Finished')
    return output_dict


if __name__ == '__main__':
    main()

