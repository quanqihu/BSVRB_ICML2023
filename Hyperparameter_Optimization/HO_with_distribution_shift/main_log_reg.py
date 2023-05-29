'''
In this project we consider hyperparameter optimization with linear model.

'''
import torch
import numpy as np
# import wandb
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import date
import argparse
import os
import pickle

from util import set_all_seeds
from datasets import Datasets
import ho_gradients


parser = argparse.ArgumentParser(description='LR_linear_ht')

### Parameter Setting
parser.add_argument('--SEED', default=123, type=int)
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--total_epoch', default=10, type=int)
parser.add_argument('--total_iter', default=1000, type=int)
parser.add_argument('--save_period', default=100, type=int)

parser.add_argument('--method', default='reg_simple', type=str)
parser.add_argument('--dataset', default='a8a', type=str, help='a1a, a8a, w8a')
parser.add_argument('--task_num', default=100, type=int)
parser.add_argument('--data_batch_size', default=32, type=int)
parser.add_argument('--task_batch_size', default=10, type=int)
parser.add_argument('--sigma_option', default='rand', type=str)
parser.add_argument('--lamb', default=1, type=float, help='regulirization parameter')
parser.add_argument('--eta', default=0.01, type=float, help='step size for p')
parser.add_argument('--alpha', default=0, type=float, help='step size for s, H')
parser.add_argument('--auto_gamma', default=1, type=int, help='whether use the gamma setting in theory')  ### Default=True, add --auto_gamma 0 to set False
parser.add_argument('--gamma', default=0.01, type=float, help='MSVR parameter')
parser.add_argument('--tau', default=0, type=float, help='step size for w')
parser.add_argument('--beta', default=0.01, type=float, help='step size for z')
parser.add_argument('--v_proj', default=False, type=bool, help='whether do projection in v updates')
parser.add_argument('--v_bound', default=0, type=float, help='v projection domain')
parser.add_argument('--v1_partialUpdate', default=False, type=bool, help='whether do partial updates for BSVRBv1')
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



def logLoss(w, x, y):
    loss = np.log(1 + np.exp((-y ) * np.dot(w, np.append([1], x))))
    return loss


def log_func(z):
    return 1 / (1 + np.exp(-z))


def log_func_der(z):
    return np.exp(z) / (1 + np.exp(z)) ** 2


def f_evaluation(w, X, Y):
    loss = 0
    data_num = len(Y)
    for j in range(len(Y)):
        loss += logLoss(w, X[j], Y[j])
    return loss/data_num


def accuracy_eval(model, X, Y):
    pred_label=np.array([])
    data_num = len(Y)
    for i in range(data_num):
        if np.dot(model, np.append([1], X[i])) >=0:
            pred_label = np.append(pred_label, 1)
        else:
            pred_label = np.append(pred_label, -1)
    return np.sum(pred_label == Y)/data_num


def grad_f(w, X, Y, data_bat_ind):
    grad_y_f = np.zeros_like(w)
    for j in data_bat_ind:
        x = np.append([1], X[j])
        y = Y[j]
        wx = np.dot(w, x)
        term1 = 1 - 1 / (1 + np.exp(-(y ) * wx))
        grad_y_f += - term1 * (y ) * x
    return grad_y_f / len(data_bat_ind)



def reg():
    m = args.task_num
    data_batch_size = args.data_batch_size
    eta = args.eta


    print('number of tasks = ', m, ', data_bat_size = ', data_batch_size, flush=True)
    print('eta = ', eta, ', SEED = ', args.SEED, flush=True)
    print('data_noise = ', args.data_noise, 'noise_perc = ', args.noise_perc, flush=True)

    ### Reproductivity
    set_all_seeds(args.SEED)

    ### Data processing
    X_train, Y_train, X_val, Y_val, X_test, Y_test, tr_num, val_num = Datasets(args.dataset, add_noise=args.data_noise, noise_perc=args.noise_perc, imb=args.imb, imb_rate=args.imb_rate)
    tr_data_ind = np.linspace(0, tr_num - 1, tr_num).astype(int)   # for training data sampling
    task_ind = np.linspace(0, m - 1, m).astype(int)   # for task sampling
    iter_per_epoch_tr = tr_num// data_batch_size

    ### linear model
    feature_num = len(X_train[0]) # number of features
    w_dim = feature_num+1

    ### Intialize variables
    w_1 = np.random.rand(w_dim)  # linear model: w^Tx+w0

    f_eval_tr_list = np.array([f_evaluation(w_1, X_train, Y_train)])
    f_eval_val_list = np.array([f_evaluation(w_1, X_val, Y_val)])
    f_eval_test_list = np.array([f_evaluation(w_1, X_test, Y_test)])
    accuracy_val_list = np.array([accuracy_eval(w_1, X_val, Y_val)])
    accuracy_test_list = np.array([accuracy_eval(w_1, X_test, Y_test)])
    best_accuracy_val = accuracy_val_list[0]
    best_accuracy_test = accuracy_test_list[0]
    best_model = w_1
    best_f_eval_val = f_eval_val_list[0]
    best_f_eval_test = f_eval_test_list[0]

    time_list = np.array([0])
    epoch_list = np.array([0])
    ite_list = np.array([0])

    epoch = 0 # count the epoch number in terms of training dataset
    iter_num = 0
    timer_start = dt.now()

    print('Start training', flush=True)

    while epoch <= args.total_epoch:


        ### sample data and task batch
        ### data shuffling after corresponding epoch ends
        tr_bat_ind = iter_num % iter_per_epoch_tr
        if tr_bat_ind == 0:
            epoch += 1
            np.random.shuffle(tr_data_ind)
        tr_bat = tr_data_ind[tr_bat_ind*data_batch_size : (tr_bat_ind+1)*data_batch_size]

        grad = grad_f(w_1,X_train,Y_train,tr_bat)
        w_1=w_1-eta * grad

        # save to log
        iter_num += 1
        if iter_num % args.eva_per == 0:
            # f_eval_tr_list = np.append(f_eval_tr_list, f_evaluation(w_1, X_train, Y_train))
            f_eval_val_list = np.append(f_eval_val_list, f_evaluation(w_1, X_val, Y_val))
            f_eval_test_list = np.append(f_eval_test_list, f_evaluation(w_1, X_test, Y_test))
            accuracy_val_list = np.append(accuracy_val_list, accuracy_eval(w_1, X_val, Y_val))
            accuracy_test_list = np.append(accuracy_test_list, accuracy_eval(w_1, X_test, Y_test))
            if accuracy_val_list[-1] > best_accuracy_val:
                best_f_eval_val = f_eval_val_list[-1]
                best_f_eval_test = f_eval_test_list[-1]
                best_accuracy_val = accuracy_val_list[-1]
                best_accuracy_test = accuracy_test_list[-1]
                best_model = w_1
            time_list=np.append(time_list,td.total_seconds(dt.now()-timer_start))
            epoch_list=np.append(epoch_list,epoch)
            ite_list=np.append(ite_list,iter_num)

             print("Epoch: {}, iter_num: {}, best_f_eval_val: {:4f}, best_f_eval_test: {:4f}, accuracy_val: {:4f}, accuracy_test: {:4f}, best_accuracy_val: {:4f}, best_accuracy_test: {:4f}, time: {:4f}".format(epoch,iter_num, best_f_eval_val, best_f_eval_test, accuracy_val_list[-1], accuracy_test_list[-1], best_accuracy_val, best_accuracy_test,time_list[-1]), flush=True)

            if np.isnan(f_eval_val_list[-1]):
                break

    # f_eval_tr_list = np.append(f_eval_tr_list, f_evaluation(w_1, X_train, Y_train))
    f_eval_val_list = np.append(f_eval_val_list, f_evaluation(w_1, X_val, Y_val))
    f_eval_test_list = np.append(f_eval_test_list, f_evaluation(w_1, X_test, Y_test))
    accuracy_val_list = np.append(accuracy_val_list, accuracy_eval(w_1, X_val, Y_val))
    accuracy_test_list = np.append(accuracy_test_list, accuracy_eval(w_1, X_test, Y_test))
    if accuracy_val_list[-1] > best_accuracy_val:
        best_f_eval_val = f_eval_val_list[-1]
        best_f_eval_test = f_eval_test_list[-1]
        best_accuracy_val = accuracy_val_list[-1]
        best_accuracy_test = accuracy_test_list[-1]
        best_model = w_1
    time_list = np.append(time_list, td.total_seconds(dt.now() - timer_start))
    epoch_list = np.append(epoch_list, epoch)
    ite_list = np.append(ite_list, iter_num)

    print("Epoch: {}, iter_num: {}, best_f_eval_val: {:4f}, best_f_eval_test: {:4f}, accuracy_val: {:4f}, accuracy_test: {:4f}, best_accuracy_val: {:4f}, best_accuracy_test: {:4f}, time: {:4f}".format(epoch,iter_num, best_f_eval_val, best_f_eval_test, accuracy_val_list[-1], accuracy_test_list[-1], best_accuracy_val, best_accuracy_test,time_list[-1]), flush=True)

    output_dict = {'f_eval_tr_list': f_eval_tr_list,
                   'f_eval_val_list': f_eval_val_list,
                   'f_eval_test_list': f_eval_test_list,
                   'time_list': time_list,
                   'epoch_list': epoch_list,
                   'ite_list': ite_list,
                   'accuracy_val_list': accuracy_val_list,
                   'accuracy_test_list': accuracy_test_list,
                   'best_model': best_model
                   }
    print('Finished')
    return output_dict


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
    sigma_path = '/sig_list.p'
    with open(sigma_path, 'rb') as handle:
        sigma_list = pickle.load(handle)
        handle.close()

    output_dict = reg()

    today = date.today()
    datestr = today.strftime("%b%d%Y")

    if args.data_noise:
        noise_flag=1
    else:
        noise_flag=0

    if args.imb:
        imb_flag=1
    else:
        imb_flag=0

    file_name = 'linearHT_{0}_{1}.p'.format(
        args.dataset, args.method)
    results_path = args.results_path

    new_file_path = os.path.join(results_path, file_name)
    with open(new_file_path, 'wb') as handle:
        pickle.dump(output_dict, handle)
        handle.close()


if __name__ == '__main__':
    main()
