import numpy as np 

def covariance_x_epsilon(size, alpha):
    epsilon = size**(-1/alpha)
    delta = 1/size
    Q = np.zeros([size,size])
    for i in range(size): 
        for j in range(i):
            Q[i,j] = np.exp(-(i-j)*(delta/epsilon))
    
    Q = Q + np.transpose(Q) 
    #print(Q)
    P = ((epsilon*(np.exp(delta/epsilon)+np.exp(-delta/epsilon)-2))/2)*Q + np.eye(size)*(delta-epsilon*(1-np.exp(-delta/epsilon)))
    return P

alpha = 0.9


for k in range(1,4):
    N = 2*10**k
    print(N, "N")
    P= covariance_x_epsilon(N, alpha)
    #print(P)
    spectral = np.linalg.norm(P, ord=2)
    print(N*(spectral**2))
    #one_norm = np.linalg.norm(Q, ord=1)
    #print(N**(1-(1/alpha))*np.exp(N**((1/alpha)-1))*N**(0.5)*spectral, "Spectral Term")