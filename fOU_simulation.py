import numpy as np 
import matplotlib.pyplot as plt 
from fOU import fOU
from estimators import *

fou = fOU(0.3, 100)

y_sample = fou.sample(100)
x_sample = x_epsilon(y_sample, 0.1, 0.3, 10)

asymptotic_estimator(x_sample,0.3, 10)
mle_estimator(x_sample, 0.3, 10)

H = 0.3
T = 10.0
SIGMA = 4.0
EPSILON = 0.2 

fou = fOU(H, T, SIGMA, EPSILON)
n_values = [400, 300, 200, 100, 75, 50, 33, 25, 20, 16,  10]

asymp_03_1_02 = []
mle_03_1_02 = []
for n in n_values:
    y_sample = fou.sample(n)
    x_sample = x_epsilon(y_sample, EPSILON, H, T)
    asymp_03_1_02.append(asymptotic_estimator(x_sample,H, T))
    mle_03_1_02.append(mle_estimator(x_sample,H, T))


print(asymp_03_1_02)
print(mle_03_1_02)
#plt.plot(np.log(asymp_03_1_02))
#plt.show()