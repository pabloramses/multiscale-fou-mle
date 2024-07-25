import numpy as np 
import seaborn as sbs
import matplotlib.pyplot as plt

matrix = np.transpose(np.load("L2_errors_H025_s4_T4.npy"))
heatmap = sbs.heatmap(matrix, cmap="jet")
heatmap.set(xlabel = r'$\alpha$', title = r'$L_2$ convergence of $\hat{\sigma}^2$ for $H=0.1$')
#heatmap.axes.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5])
heatmap.axes.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5, 9.5])
#heatmap.axes.set_xticklabels(['',0.25,'',0.35,'',0.45,'',0.55,'', 0.65, '', 0.75, '', 0.85, '', 0.95])
heatmap.axes.set_xticklabels(['',0.2,'',0.4,'',0.6, '2/3',0.7,'',0.9])
#heatmap.axes.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
heatmap.axes.set_yticks([0,1,2,3,4])
#epsilon = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
epsilon = [0.001, 0.005, 0.01, 0.05, 0.1]
heatmap.axes.set_yticklabels(epsilon, rotation=0)

#plt.xlabel(r'$\epsilon$')
#plt.ylabel(r'$\overline{\tilde{\sigma}}^{10}$')
#plt.title(r'Mean $\tilde{\sigma}$ as $\epsilon\rightarrow 0$ with $\delta=\epsilon^{\alpha}$ and \textbf{H=0.8}')
plt.ylabel(r'$\epsilon$', rotation=0)
plt.savefig('L2_convergence_025_s4_T4',bbox_inches='tight')
plt.show()
ALPHA = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 2/3, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 
