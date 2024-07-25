import numpy as np 
import matplotlib.pyplot as plt 
from estimators import * 

H = np.array([0.1 , 0.2, 0.3, 0.4, 0.5])
T = 1

sizes = [10, 50, 100, 1000, 3000, 5000]
sizes_string = ["10", "50", "100", "1000", "3000","5000"] #["5", "10", "50", "100", "500", "1000"]
values = []
for h in H:
    for N in sizes:
        DELTA = T/N
        unnormalised_covmatrix = covariance_matrix_fmb(h, N)
        unnormalised_inverse = np.linalg.inv(unnormalised_covmatrix)

        #normalised_covmatrix = (DELTA**(2*h))*unnormalised_covmatrix
        normalised_inverse = (1/(DELTA**(2*h)))*unnormalised_inverse

        #eigen_covmatrix = np.linalg.eigvals(normalised_covmatrix)
        #print("N equals ", N, "\n")
        #print(eigen_covmatrix)
        spectrum_inverse = np.linalg.norm(normalised_inverse,2)
        #spectrum = np.linalg.norm(unnormalised_covmatrix,2)
        values.append(spectrum_inverse)

#print(values)
beta = 2-2*H
plt.plot(np.log(values[0:6])-beta[0]*np.log(sizes), color = 'red')
plt.plot(np.log(values[6:12])-beta[1]*np.log(sizes), color = 'blue')
plt.plot(np.log(values[12:18])-beta[2]*np.log(sizes), color = 'green')
plt.plot(np.log(values[18:24])-beta[3]*np.log(sizes), color = 'chocolate')
plt.plot(np.log(values[24:30])-beta[4]*np.log(sizes), color = 'black')
plt.xticks([0,1,2,3,4,5], sizes_string)
plt.xlabel('n')
plt.ylabel(r'$\log (||P^{-1}||_2)-\log (n)$')
plt.legend(bbox_to_anchor = (1.04,1),labels = ['H=0.1','H=0.2', 'H=0.3','H=0.4', 'H=0.5'])
plt.savefig('matrix_order',bbox_inches='tight')
plt.close()
#print(spectrum_cov)
#print(spectrum_inverse)
#print(1/np.linalg.norm(normalised_covmatrix, -2))