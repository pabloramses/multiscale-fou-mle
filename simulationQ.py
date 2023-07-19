import numpy as np 

def Q_matrix(N, alpha):
    Q = np.zeros([N-1,N-1])
    for i in range(N-1): 
        for j in range(i):
            Q[i,j] = np.exp(-(i-j)*(N**(1/alpha-1)))
    
    return Q + np.transpose(Q) 


Q = Q_matrix(5,0.8)


alpha = 0.8

for k in range(1,4):
    N = 2*10**k
    Q = Q_matrix(N, alpha)
    spectral = np.linalg.norm(Q, ord=2)
    #one_norm = np.linalg.norm(Q, ord=1)
    print(N**(1-(1/alpha))*np.exp(N**((1/alpha)-1))*N**(0.5)*spectral, "Spectral Term")
    #print("spectral is ", spectral)
    #print("1-norm is ", one_norm)
    #print(np.linalg.norm(Q), "Frobenius")

