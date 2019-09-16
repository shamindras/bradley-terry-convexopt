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
    data: TxNxN array
    lambdas_smooth: a vector of query lambdas
    opt_fn: a python function in a particular form of 
        opt_fn(data, lambda_smooth, beta_init=None, **kwargs)
        kwargs might contain hyperparameters 
        (e.g., step size, max iteration, etc.) for
        the optimization function
    num_loocv: the number of random samples left-one-out cv sample
    get_estimate: whether or not we calculate estimates beta's for 
        every lambdas_smooth. If True, we use those estimates as 
        initial values for optimizations with cv data
    **kwargs: keyword arguments for opt_fn
    ----------
    Output:
    lambda_cv: lambda_smooth chosen after cross-validation
    nll_cv: average cross-validated negative loglikelihood 
    beta_cv: beta chosen after cross-validation. None if get_estimate is False
    '''    
    lambdas_smooth = lambdas_smooth.flatten()
    betas = [None] * lambdas_smooth.shape[0]
    
    for i, lambda_smooth in enumerate(lambdas_smooth):
        _, beta = opt_fn(data, lambda_smooth, **kwargs)
        betas[i] = beta.reshape(data.shape[:2])
        
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    console = open('/dev/stdout', 'w')
    loglikes_loocv = np.zeros(lambdas_smooth.shape)
    for i in range(num_loocv):
        data_loocv = data.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loocv[tuple(rand_index)] -= 1

        for j, lambda_smooth in enumerate(lambdas_smooth):
            _, beta_loocv = opt_fn(data_loocv, lambda_smooth, 
                                   beta_init = betas[j], **kwargs)
            beta_loocv = beta_loocv.reshape(data.shape[:2])
            loglikes_loocv[j] += beta_loocv[rand_index[0],rand_index[1]] \
                   - np.log(np.exp(beta_loocv[rand_index[0],rand_index[1]])
                            + np.exp(beta_loocv[rand_index[0],rand_index[2]]))

        console.write("%d-th cv done\n"%(i+1))
        console.flush()
        
    return (lambdas_smooth[np.argmax(loglikes_loocv)], -loglikes_loocv/num_loocv, 
            betas[np.argmax(loglikes_loocv)])