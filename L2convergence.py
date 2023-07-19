import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *

T = 1
H = 0.5

epsilon = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
first = 0
ALPHA = 0.2
for a in range(16):
    row = np.array([])
    print("ALPHA: ", ALPHA)
    for EPSILON in epsilon: 
        for rep in range(100):
            N = int(T/(EPSILON**2))

            fou = fOU(H, T, epsilon=EPSILON)

            y_sample1 = fou.sample(N)
            x_sample1 = x_epsilon(y_sample1, EPSILON, H, T)
            x_subsample1 = subsample(x_sample1, EPSILON, T, ALPHA)
            mle = mle_estimator(x_subsample1, H, T)
                    
            if rep==0: 
                mean_mse = (1/100)*(mle-1)**2
                
            else:
                mean_mse = mean_mse + (1/100)*(mle-1)**2
                

                    
                    #print("alpha ", alpha, " done")
        print("EPS", EPSILON, " done")
        print("mean mse ", mean_mse)
        row = np.append(row, mean_mse)
    if first==0:
        matrix = np.array(row)
        first = 1
    else: 
        matrix = np.vstack((matrix,row))
    ALPHA += 0.05


np.save('L2_errors.npy', matrix)