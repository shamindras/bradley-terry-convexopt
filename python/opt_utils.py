import numpy as np
import scipy as sc
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

def PGD_l2_sq(data, lambda_smooth, beta_init=None,
              max_iter=1000, ths=1e-12, 
              step_size=0.1, max_back=100, a=0.2, b=0.5,
              verbose=False):
    '''
    conduct a proximal gradient descent of the model
    ----------
    Input:
    data: TxNxN array
    lambda_smooth: numeric
    beta_init: TxN array or TN vector
    max_iter, max_back: integer
    ths, step_size, a, b: numeric
    verbose: logic
    ----------
    Output:
    beta: optimization solution
    objective_wback: history of objective during optimization
    '''
        
    # intiialize optimization
    if beta_init is None:
        beta = np.zeros(data.shape[:2])
    else:
        beta = np.reshape(beta_init, data.shape[:2])
    nll = model.neg_log_like(beta, data)

    # initialize record
    objective_wback = [opt_utils.objective_l2_sq(beta, data, lambda_smooth)]
    if verbose:
        print("initial objective value: %f"%objective_wback[-1])

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = model.grad_nl(beta, data).reshape(beta.shape)

        # backtracking line search
        s = step_size
        for j in range(max_back):
            beta_new = opt_utils.prox_l2_sq(beta - s*gradient, s, lambda_smooth)
            beta_diff = beta_new - beta

            nll_new = model.neg_log_like(beta_new, data)
            nll_back = (nll + np.sum(gradient * beta_diff) 
                        + np.sum(np.square(beta_diff)) / (2*s))

            if nll_new <= nll_back:
                break
            s *= b

        # proximal gradient update
        beta = beta_new
        nll = nll_new

        # record objective value
        objective_wback.append(opt_utils.objective_l2_sq(beta, data, lambda_smooth))
        
        if verbose:
            print("%d-th PGD, objective value: %f"%(i+1, objective_wback[-1]))
        if objective_wback[-2] - objective_wback[-1] < ths:
            if verbose:
                print("Converged!")
            break

    if i >= max_iter:
        if verbose:
            print("Not converged.")
    
    return beta, objective_wback

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