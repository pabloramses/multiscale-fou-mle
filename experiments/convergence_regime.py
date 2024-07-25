import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *
import multiprocessing

T = 1
EPSILON = 0.005
SIGMA = 1.0
ALPHA = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 2/3, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] 

def tar(alpha):
    row = [alpha]
    H = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 2/3, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 
    for h in H: 
        for rep in range(100):
            N = int(T/(EPSILON**2))

            fou = fOU(h, T, SIGMA, epsilon=EPSILON)

            y_sample1 = fou.sample(N)
            x_sample1 = x_epsilon(y_sample1, EPSILON, h, T)
            x_subsample1 = subsample(x_sample1, EPSILON, T, alpha)
            mle = mle_estimator(x_subsample1, h, T)
                        
            if rep==0: 
                mean_mse = (1/100)*(mle-SIGMA**2)**2
                    
            else:
                mean_mse = mean_mse + (1/100)*(mle-SIGMA**2)**2
                    

                        

        print("H", h, " done")
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
    np.save('L2_errors_eps005.npy', results) 