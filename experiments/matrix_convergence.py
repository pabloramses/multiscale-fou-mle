import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *

T = 1
H = 0.5

EPSILON = 0.001
#epsilon = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
ALPHA = 2/3

#for EPSILON in epsilon: 

reps = 100
for rep in range(reps):
    N = int(T/(EPSILON**2))

    fou = fOU(H, T, epsilon=EPSILON)

    y_sample1 = fou.sample(N)
    x_sample1 = x_epsilon(y_sample1, EPSILON, H, T)
    x_subsample1 = np.array(subsample(x_sample1, EPSILON, T, ALPHA))
    n = x_subsample1.shape[0]
    x_increments = x_subsample1[1:] - x_subsample1[:n-1]
    if rep==0: 
        sample_x_eps = np.array(x_increments)
    else:
        sample_x_eps = np.vstack((sample_x_eps, x_increments)) 

estimated_cov_x_eps = np.matmul(sample_x_eps.T, sample_x_eps)/(reps-1)
fbm_cov = ((EPSILON**ALPHA)**(2*H-1))*covariance_matrix_fmb(H, n-1)

estimated_cov_x_eps_inv = ((EPSILON**ALPHA))*np.linalg.inv(estimated_cov_x_eps)
fbm_cov_inv = np.linalg.inv(fbm_cov)

spectral_norm = np.linalg.norm(estimated_cov_x_eps_inv-fbm_cov_inv, ord=2)
print(spectral_norm)