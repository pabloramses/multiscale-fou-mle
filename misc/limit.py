import numpy as np 


"""
#N = 100000
H = 0.3

sums = []
for p in range(1,10):
    N = 10**p
    counter = 0.0
    for k in range(1,N-1):
        counter += 2*(N-1-k)*((((k+1)**2)**H + ((k-1)**2)**H - 2*k**(2*H))**2)

    sums.append((1/N)**(4*H)*counter)
print(sums)
"""
def rho_fBM(i, j, h):
    return 0.5*( 
                (np.abs(i-j)+1)**(2*h) + 
                (np.abs(i-j)-1)**(2*h) -
                2*(np.abs(i-j))**(2*h)
               )

def matrix(h, n):
    sigma_n = np.zeros([n,n])
    for i in range(n):
        for j in range(i):
            sigma_n[i,j] = rho_fBM(i,j,h)
    return sigma_n + np.transpose(sigma_n) 

n = 1000
H = 0.75
mat = ((1/n)**(2*H))*matrix(0.9, n)
ev = np.linalg.eig(mat)
print(np.min(np.abs(ev[0])))
print(n**(-2*H))