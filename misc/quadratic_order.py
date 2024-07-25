import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *

epsilon = [0.1, 0.05, 0.01, 0.005, 0.0025, 0.001]
alpha = 2/3
delta = np.array(epsilon)**alpha
T = 1
SIGMA = 1
H = 0.75


mean_values = np.array([])
for EPSILON in epsilon:
    N = int(T/(EPSILON**2))
    quad_var = np.array([])
    for rep in range(100):
        fou = fOU(H, T, SIGMA, epsilon=EPSILON)

        y_sample = fou.sample(N)
        x_sample = x_epsilon(y_sample, EPSILON, H, T)
        x_subsample = np.array(subsample(x_sample, EPSILON, T, alpha))
        length = x_subsample.shape[0]
        x_increments = x_subsample[1:] - x_subsample[:length-1]
        quad_var = np.append(quad_var, np.sum(np.array(x_increments**2))**2)

    mean_values = np.append(mean_values, np.sqrt(np.mean(quad_var)))

plt.plot(mean_values, color="blue")
plt.plot(delta**(2*H-1), color = "red")
plt.xticks([0,1,2,3,4,5], np.round(delta,3))
plt.xlabel(r'$\delta$')
plt.ylabel(' ')
plt.title('H=3/4')
plt.legend(bbox_to_anchor = (1,1),labels = [r'$||\langle X^{\epsilon}\rangle_{\delta}||_{L_2}$', r'$\delta^{2H-1}$'])
plt.savefig('QV_order',bbox_inches='tight')
plt.close()