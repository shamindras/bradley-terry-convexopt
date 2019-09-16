import pandas as pd
import scipy as sc
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

def beta_gaussian_process(N, T, mu_parameters, cov_parameters, mu_type = 'constant', cov_type = 'toeplitz'):
    '''
    generate beta via a Gaussian process
    '''
    if mu_type == 'constant':
        loc, scale = mu_parameters
        mu_start = stats.norm.rvs(loc = loc,scale = scale,size = N,random_state = 100)
        mu = [np.ones(T) * mu_start[i] for i in range(N)]
    if cov_type == 'toeplitz':
        alpha, r = cov_parameters
    ##### strong auto-correlation case, off diagonal  = 1 - T^(-alpha) * |i - j|^r
    off_diag = 1 - T ** (-alpha) * np.arange(1,T + 1) ** r
    cov_single_path = sc.linalg.toeplitz(off_diag,off_diag)

    return np.array([np.random.multivariate_normal(mean = mu[i],cov = cov_single_path,size = 1).ravel() for i in range(N)]).T


def get_game_matrix_list(N,T,tn,beta):
    '''
    get the list of T game matrices
    -------------
    Input:
    N: number of teams
    T: number of seasons
    tn: list of number of games between each pair of teams
    beta: a T-by-N array
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
                pij = np.exp(beta[t,i] - beta[t,j]) / (1 + np.exp(beta[t,i] - beta[t,j]))
                nij = np.random.binomial(n = tn[ind], p = pij, size = 1)
                game_matrix[i,j],game_matrix[j,i] = nij, tn[ind] - nij
        game_matrix_list[t] = game_matrix
    return np.array(game_matrix_list)

def beta_markov_chain(num_team,num_season,var_latent = 1,coef_latent = 1,sig_latent = 1,draw = True):
    pc_latent = diags([-coef_latent/var_latent, 1/var_latent, -coef_latent/var_latent],
                      offsets = [-1, 0, 1],
                      shape=(num_season, num_season)).todense()
    pc_latent[np.arange(num_season-1), np.arange(num_season-1)] += (coef_latent**2)/var_latent
    var_latent = np.linalg.inv(pc_latent)
    inv_sqrt_var = np.diag(1/np.sqrt(np.diag(var_latent)))
    var_latent = sig_latent * inv_sqrt_var @ var_latent @ inv_sqrt_var
    
    latent = np.transpose(
        np.random.multivariate_normal(
            [0]*num_season, var_latent, num_team))
    
    if draw:
        f = plt.figure(1, figsize = (12,5))
        
        ax = plt.subplot(121)
        plt.imshow(var_latent, 
           cmap='RdBu', vmin=-sig_latent, vmax=sig_latent)
        plt.colorbar()
        plt.title("variance")
        
        ax = plt.subplot(122)
        plt.imshow(latent, cmap='RdBu',
           vmin=-np.max(np.abs(latent)), 
           vmax=np.max(np.abs(latent)))
        plt.xlabel("team number")
        plt.ylabel("season number")
        plt.title("beta")
    return latent
'''
some examles of running functions beta_gaussian_process and get_game_matrix_list
##### example of generating

N = 10 # number of teams
T = 10 # number of seasons/rounds/years
tn_median = 100
bound_games = [tn_median - 2,tn_median + 2] # bounds for the number of games between each pair of teams

##### tn: list of number of games between each pair of teams
tn = stats.randint.rvs(low = int(bound_games[0]), high = int(bound_games[1]), size = int(T * N * (N - 1) / 2))

##### get beta here #####
beta = beta_gaussian_process(N, T, mu_parameters = [0,1], cov_parameters = [alpha,r], mu_type = 'constant', cov_type = 'toeplitz')

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


