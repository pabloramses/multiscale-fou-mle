import numpy as np 
import matplotlib.pyplot as plt 

EPSILON = np.array([0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001])
DELTA = EPSILON**(2/3)
H = 0.75 

orders = np.load('remainder_orders.npy')

#plt.plot((EPSILON**(2*H))*orders[:,0])
plt.plot(orders[:,2])
#plt.plot((EPSILON**(H))*orders[:,2])
#plt.plot(orders[:,3])
plt.plot((DELTA**(1-H)), color="red")
#plt.legend(bbox_to_anchor = (1.04,1),labels = ['yy','by', 'yb','sum', 'ratio'])
plt.show()