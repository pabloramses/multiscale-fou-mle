import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *
import multiprocessing

T = 4
H = 0.25
SIGMA = 4.0
ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2/3, 0.7, 0.8, 0.9] 

def tar(alpha):
    print(alpha)
    row = [alpha]
    epsilon = [0.001, 0.005, 0.01, 0.05, 0.1]
    for EPSILON in epsilon: 
        for rep in range(100):
            N = int(T/(EPSILON**2))

            fou = fOU(H, T, SIGMA, epsilon=EPSILON)

            y_sample1 = fou.sample(N)
            x_sample1 = x_epsilon(y_sample1, EPSILON, H, T)
            x_subsample1 = subsample(x_sample1, EPSILON, T, alpha)
            mle = mle_estimator(x_subsample1, H, T)
                        
            if rep==0: 
                mean_mse = (1/100)*(mle-SIGMA**2)**2
                    
            else:
                mean_mse = mean_mse + (1/100)*(mle-SIGMA**2)**2
                    

                        

        print("EPS", EPSILON, " done")
        print("mean mse ", mean_mse)
        row.append(mean_mse)
    return np.array(row) 

if __name__ == '__main__':

    with multiprocessing.Pool() as pool: 
       p = pool.map(tar, ALPHA)
       one = True
       for i in p: 
           if one: 
               results = i
               one = False
           else: 
               results = np.vstack((results, i))

    print(results)
    np.save('L2_errors_H025_s4_T4.npy', results) 