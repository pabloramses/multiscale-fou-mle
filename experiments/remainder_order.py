import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *
import multiprocessing

T = 5
H = 0.75
SIGMA = 1.0
ALPHA = 2/3
EPSILON = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]

def tar(epsilon):
    for rep in range(100):
        N = int(T/(epsilon**2))
        fou = fOU(H, T, SIGMA, epsilon=epsilon)

        y_sample = np.array(fou.sample(N))
        x_sample = x_epsilon(y_sample, epsilon, H, T)
        x_subsample = np.array(subsample(x_sample, epsilon, T, ALPHA))
        x_delta = x_subsample[1:] - x_subsample[:np.shape(x_subsample)[0]-1]
        x_covariance_matrix = np.matmul(np.transpose(np.matrix(x_delta)), np.matrix(x_delta))
        P_n = covariance_matrix_fmb(H, np.shape(x_subsample)[0]-1)*(epsilon**(2*ALPHA))
                        
        if rep==0: 
            mean_covariance_x= (1/100)*x_covariance_matrix
                    
        else:
            mean_covariance_x = mean_covariance_x + (1/100)*x_covariance_matrix
    result = np.linalg.norm(mean_covariance_x - P_n,1)
    return result


if __name__ == '__main__':

    with multiprocessing.Pool() as pool: 
       p = pool.map(tar, EPSILON)
       one = True
       for i in p: 
           if one: 
               results = i
               one = False
           else: 
               results = np.vstack((results, i))

    np.save('remainder_order_direct.npy', results) 

