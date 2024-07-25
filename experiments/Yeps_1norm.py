import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *
import multiprocessing

T = 1
H = 0.75
SIGMA = 1.0
ALPHA = 2/3
EPSILON = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]

def tar(epsilon):
    #row = [alpha]
    for rep in range(100):
        N = int(T/(epsilon**2))

        fou = fOU(H, T, SIGMA, epsilon=epsilon)

        y_sample = np.array(fou.sample(N))
        driving_fbm = fou.give_noise()
        y_subsample = np.array(subsample(y_sample, epsilon, T, ALPHA))
        driving_fmb_subsample = np.array(subsample(driving_fbm, epsilon, T, ALPHA))
        y_delta = y_subsample[1:] - y_subsample[:np.shape(y_subsample)[0]-1]
        b_delta = driving_fmb_subsample[1:] - driving_fmb_subsample[:np.shape(driving_fmb_subsample)[0]-1]
        y_covariance_matrix = np.matmul(np.transpose(np.matrix(y_delta)), np.matrix(y_delta))
        by_matrix = np.matmul(np.transpose(np.matrix(b_delta)), np.matrix(y_delta))
        yb_matrix = np.matmul(np.transpose(np.matrix(y_delta)), np.matrix(b_delta))
            
                        
        if rep==0: 
            mean_covariance_y= (1/100)*y_covariance_matrix
            mean_by= (1/100)*by_matrix
            mean_yb= (1/100)*yb_matrix
                    
        else:
            mean_covariance_y = mean_covariance_y + (1/100)*y_covariance_matrix
            mean_by = mean_by + (1/100)*by_matrix
            mean_yb = mean_yb + (1/100)*yb_matrix

    return np.linalg.norm(mean_covariance_y,1), np.linalg.norm(mean_by,1), np.linalg.norm(mean_yb,1), np.linalg.norm(epsilon**(2*H)*mean_covariance_y + epsilon**(H)*mean_yb + epsilon**(H)*mean_by,1)


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

    np.save('remainder_orders.npy', results) 

