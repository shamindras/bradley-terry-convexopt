def loglike(beta,game_matrix_list):
    '''
    compute the negative loglikelihood
    ------------
    Input:
    beta: can be a T-by-N array or a T*N-by-1 array
    game_matrix_list: records of games, T-by-N-by-N array
    ------------
    -l: negative loglikelihood, a number
    '''
    # beta could be a T-by-N matrix or a T*N-length vector
    shape = beta.shape
    T, N = game_matrix_list.shape[0:2]
    if len(shape) == 1:
        beta = beta.reshape(T,N)
    # l stores the loglikelihood
    l = 0
    N_one = np.ones(N).reshape(N,1)
    for t in range(T):
        Cu = np.triu(game_matrix_list[t]) # equivalent to [t,:,:]
        Cl = np.tril(game_matrix_list[t])
        b = beta[t,:].reshape(N,1)
        D = b @ N_one.T - N_one @ b.T
        W = np.log(1 + np.exp(D))
        l += N_one.T @ (Cu * D) @ N_one - N_one.T @ ((Cu + Cl.T) * np.triu(W)) @ N_one
    return - l[0,0]


def grad_l(beta,game_matrix_list):
    '''
    compute the gradient of the negative loglikelihood
    ------------
    Input:
    beta: can be a T-by-N array or a T*N-by-1 array
    game_matrix_list: records of games, T-by-N-by-N array
    ------------
    Output:
    -grad: gradient of negative loglikelihood, a T*N-by-1 array
    '''
    # beta could be a T-by-N array or a T*N-length vector
    shape = beta.shape
    T, N = game_matrix_list.shape[0:2]
    if len(shape) == 1:
        beta = beta.reshape(T,N)
    # g stores the gradient
    g = np.zeros(N * T).reshape(T,N)
    N_one = np.ones(N).reshape(N,1)
    for t in range(T):
        C = game_matrix_list[t]
        b = beta[t,:].reshape(N,1)
        W = np.exp(b @ N_one.T) + np.exp(N_one @ b.T)
        g[t,:] = ((C / W) @ np.exp(b) - (C / W).T @ N_one * np.exp(b)).ravel()
    return - g.reshape(N * T,1)

def Hess_l(beta,game_matrix_list):
    '''
    compute the Hessian of the negative loglikelihood
    ------------
    Input:
    beta: can be a T-by-N array or a T*N-by-1 array
    game_matrix_list: records of games, T-by-N-by-N array
    ------------
    Output:
    -H: Hessian of negative loglikelihood T*N-by-T*N array
    '''
    # beta could be a T-by-N array or a T*N-length array
    shape = beta.shape
    T, N = game_matrix_list.shape[0:2]
    if len(shape) == 1:
        beta = beta.reshape(T,N)
    # g stores the gradient
    H = np.zeros(N ** 2 * T ** 2).reshape(T * N,T * N)
    N_one = np.ones(N).reshape(N,1)
    for t in range(T):
        Cu = np.triu(game_matrix_list[t]) # equivalent to [t,:,:]
        Cl = np.tril(game_matrix_list[t])
        Tm = Cu + Cl.T + Cu.T + Cl
        b = beta[t,:].reshape(N,1)
        W = np.exp(b @ N_one.T) + np.exp(N_one @ b.T)
        H_t = Tm * np.exp(b @ N_one.T + N_one @ b.T) / W ** 2
        H_t += -np.diag(sum(H_t))
        ind = range(t * N, (t + 1) * N)
        H[t * N:(t + 1) * N,t * N:(t + 1) * N] = H_t
    return - H