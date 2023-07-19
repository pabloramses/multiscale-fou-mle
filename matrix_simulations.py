import numpy as np 
from estimators import *

P_eps_inv = covariance_x_epsilon(0.01, 0.1, 5)
print(P_eps_inv)
print(np.linalg.inv(P_eps_inv))