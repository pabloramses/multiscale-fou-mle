import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU
from estimators import *

T = 1
EPSILON = 0.01
N = int(T/(EPSILON**2))
H = 0.5
ALPHA = 0.9
print("Delta", EPSILON**ALPHA)
t_grid = np.linspace(0,T,N+2)

fou = fOU(H, T, epsilon=EPSILON)

y_sample = fou.sample(N)
x_sample = x_epsilon(y_sample, EPSILON, H, T)
x_subsample = subsample(x_sample, EPSILON, T, ALPHA)
print(len(x_subsample))
mle = mle_estimator(x_subsample, H, T)
print(mle)
