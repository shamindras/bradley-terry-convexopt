import numpy as np
import scipy as sc
import scipy.linalg as spl
import grad_utils as model

def loocv(data, lambdas_smooth, opt_fn,
          num_loocv = 200, get_estimate = True, **kwargs):
    '''
    conduct local
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    lambda_cv: lambda_smooth chosen after cross-validation
    nll_cv: average cross-validated negative loglikelihood 
    beta_cv: beta chosen after cross-validation. None if get_estimate is False
    '''    
    lambdas_smooth = lambdas_smooth.flatten()
    betas = [None] * lambdas_smooth.shape[0]
    
    for i, lambda_smooth in enumerate(lambdas_smooth):
        beta, _ = opt_fn(data, lambda_smooth, **kwargs)
        betas[i] = beta
        
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    loglikes_loocv = np.zeros(lambdas_smooth.shape)
    for i in range(num_loocv):
        data_loocv = data.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loocv[tuple(rand_index)] -= 1

        for j, lambda_smooth in enumerate(lambdas_smooth):
            beta_loocv, _ = opt_fn(data_loocv, lambda_smooth, 
                                   beta_init = betas[j], **kwargs)
            loglikes_loocv[j] += beta_loocv[rand_index[0],rand_index[1]] \
                   - np.log(np.exp(beta_loocv[rand_index[0],rand_index[1]])
                            + np.exp(beta_loocv[rand_index[0],rand_index[2]]))

        print("%d-th cv done"%(i+1))
        
    return (lambdas_smooth[np.argmax(loglikes_loocv)], -loglikes_loocv/num_loocv, 
            betas[np.argmax(loglikes_loocv)])