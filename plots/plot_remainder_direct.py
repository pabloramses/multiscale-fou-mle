import numpy as np 
import matplotlib.pyplot as plt 

EPSILON = np.array([0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001])
DELTA = EPSILON**(2/3)
H = 0.75 

orders = np.load('remainder_order_direct.npy')
print(np.transpose(orders)/((EPSILON/DELTA)**(2*H)))
plt.plot(np.transpose(np.transpose(orders)/((EPSILON/DELTA)**(2*H))), color="red")
#plt.legend(bbox_to_anchor = (1.04,1),labels = ['yy','by', 'yb','sum', 'ratio'])
plt.show()