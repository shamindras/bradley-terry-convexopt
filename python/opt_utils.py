import numpy as np
import scipy as sc
import pandas as pd
import scipy.linalg as spl
import grad_utils as model


########################## squared l2 penalty ############################


def objective_l2_sq(beta, game_matrix_list, l_penalty):
    '''
    compute the objective of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_penalty = np.sum(np.square(beta[:-1]-beta[1:]))
    
    return model.neg_log_like(beta, game_matrix_list) + l_penalty * l2_penalty


def grad_l2_sq(beta, game_matrix_list, l):
    '''
    compute the gradient of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_grad = model.grad_nl(beta, game_matrix_list)
    l2_grad[N:] += l * 2 * ((beta[1:]-beta[:-1])).reshape(((T - 1) * N, 1))
    l2_grad[:-N] += l * 2 *((beta[:-1]-beta[1:])).reshape(((T - 1) * N, 1))
    
    return  l2_grad


def hess_l2_sq(beta, game_matrix_list, l):
    '''
    compute the Hessian of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_hess = model.hess_nl(beta, game_matrix_list)
    off_diag = np.array([2] + [0] * (N - 1) + [-1] + [0] * (N * (T - 1) - 1))
    l2_hess += l * 2 * sc.linalg.toeplitz(off_diag,off_diag)
    l2_hess[0:N,0:N] -= l * 2 * np.diag(np.ones(N))
    l2_hess[-N:,-N:] -= l * 2 * np.diag(np.ones(N))
    return  l2_hess




########################## l2 penalty ############################
def objective_l2(beta, game_matrix_list, l_penalty):
    '''
    compute the objective of the model (neg_log_like + l2)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    diff = beta[:-1]-beta[1:]
    l2_penalty = sum([np.linalg.norm(diff[i]) for i in range(T - 1)])
    
    return model.neg_log_like(beta, game_matrix_list) + l_penalty * l2_penalty


def grad_l2(beta, game_matrix_list, l):
    '''
    compute the gradient of the model (neg_log_like + l2)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_grad = model.grad_nl(beta, game_matrix_list)
    diff = beta[1:] - beta[:-1]
    w = np.array([np.linalg.norm(diff[i]) for i in range(T - 1)])
    w[w != 0] = 1 / w[w != 0]
    l2_grad[N:] += l * np.array([diff[i] * w[i] for i in range(T - 1)]).reshape(((T - 1) * N, 1))
    
    diff = beta[:-1] - beta[1:]
    w = np.array([np.linalg.norm(diff[i]) for i in range(T - 1)])
    w[w != 0] = 1 / w[w != 0]
    l2_grad[:-N] += l * np.array([diff[i] * w[i] for i in range(T - 1)]).reshape(((T - 1) * N, 1))
    
    return  l2_grad