import numpy as np 
import matplotlib.pyplot as plt 

results = np.load("results.npy")
plt.plot(results)
plt.show()