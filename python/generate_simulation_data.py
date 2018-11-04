import pandas as pd
import scipy as sc
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def beta_gaussian_process(N, T, mu_parameters, cov_parameters, mu_type = 'constant', cov_type = 'toeplitz'):
    '''
    generate beta via a Gaussian process
    '''
    beta_mu = stats.norm.rvs(loc = 0,scale = 1,size = N,random_state = 100)
    if mu_type == 'constant':
        mu_start = mu_parameters[0]
        mu = [np.ones(T) * mu_start[i] for i in range(N)]
    if cov_type == 'toeplitz':
        alpha, r = cov_parameters
    ##### strong auto-correlation case, off diagonal  = 1 - T^(-alpha) * |i - j|^r
    off_diag = 1 - T ** (-alpha) * np.arange(1,T + 1) ** r
    cov_single_path = sc.linalg.toeplitz(off_diag,off_diag)

    return [np.random.multivariate_normal(mean = mu[i],cov = cov_single_path,size = 1).ravel() for i in range(N)]


def get_game_matrix_list(N,T,tn,beta):
    '''
    get the list of T game matrices
    -------------
    Input:
    N: number of teams
    T: number of seasons
    tn: list of number of games between each pair of teams
    beta: a list of T sublist, each storing the beta for N teams
    -------------
    Output:
    game_matrix_list: a 3-d np.array of T game matrices, each matrix (t,:,:) stores the number of times that i wins j at entry (i,j) at season t
    '''
    game_matrix_list = [None] * T
    ind = -1 # a counter used to get tn
    for t in range(T):
        game_matrix = np.zeros(N * N).reshape(N,N)
        for i in range(N):
            for j in range(i + 1,N):
                ind += 1
                pij = np.exp(beta[t][i] - beta[t][j]) / (1 + np.exp(beta[t][i] - beta[t][j]))
                nij = np.random.binomial(n = tn[ind], p = pij, size = 1)
                game_matrix[i,j],game_matrix[j,i] = nij, tn[ind] - nij
        game_matrix_list[t] = game_matrix
    return np.array(game_matrix_list)


'''
some examles of running Wanshan's functions
##### example of generating


N = 10 # number of teams
T = 10 # number of seasons/rounds/years
bound_games = [8,12] # bounds for the number of games between each pair of teams

##### tn: list of number of games between each pair of teams
tn = stats.randint.rvs(low = int(bound_games[0]), high = int(bound_games[1]), size = int(T * N * (N - 1) / 2))

##### get beta here #####
beta = beta_gaussian_process(N, T, mu_parameters = [beta_mu], cov_parameters = [alpha,r], mu_type = 'constant', cov_type = 'toeplitz')

game_matrix_list = get_game_matrix_list(N,T,tn,beta)



##### example of drawing paths

beta = beta_gaussian_process(N, T, mu_parameters = [beta_mu], cov_parameters = [alpha,r], mu_type = 'constant', cov_type = 'toeplitz')

f = plt.figure(3, figsize = (9,4.5))
ax = plt.subplot(111)
for i in range(T):
    ax.plot(range(1,T + 1),beta[i],c=np.random.rand(3,1),marker = '.',label = 'Team' + str(i),linewidth=1)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
'''


