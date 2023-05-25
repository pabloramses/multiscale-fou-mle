import numpy as np 
import matplotlib.pyplot as plt 
from fOU import fOU
from estimators import *


T = 10
H = 0.8
epsilon = np.linspace(0.001, 0.1, 10)

for rep in range(10):
    for a in range(1,10):
        row =[]
        alpha = 0.1*a
        for EPSILON in epsilon: 
            N = int(T/(EPSILON**2))

            fou = fOU(H, T, epsilon=EPSILON)

            y_sample1 = fou.sample(N)
            x_sample1 = x_epsilon(y_sample1, EPSILON, H, T)
            x_subsample1 = subsample(x_sample1, EPSILON, T, alpha)
            row.append(mle_estimator(x_subsample1, H, T))
            #print(EPSILON, " done")
        if a==1: 
            mle = np.array(row)
        else:
            mle = np.vstack((mle, row)) 

        
        #print("alpha ", alpha, " done")
    if rep ==0:
        mean_mle = (1/10)*mle 
    else: 
        mean_mle = mean_mle + (1/10)*mle
    print("rep ", rep, " done")
np.save('results08.npy', mean_mle)
print(mean_mle)
#plt.plot(x_sample1)
#plt.plot(x_subsample1)
#plt.plot(x_subsample1)
#plt.show()
#rem = reminder_approx(x_sample, EPSILON, 1)
#print(rem)