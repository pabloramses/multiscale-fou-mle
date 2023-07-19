import numpy as np 
from scipy.special import gamma 

def rho_fBM(i, j, h):
    """
    Assuming that we have an equispaced grid of points with distance 1 between each two this function computes the correlation of
    the incrrements of an h-fractional brownian motion observed in each pair of points
    """
    return 0.5*( 
                (np.abs(i-j)+1)**(2*h) + 
                (np.abs(i-j)-1)**(2*h) -
                2*(np.abs(i-j))**(2*h)
               )

def covariance_matrix_fmb(h, n):
    """
    Computes the covariance matrix of a set of increments observed on a discretised fractional Brownian Motion observed in an
    equispaced grid with diameter 1
    """
    sigma_n = np.zeros([n,n])
    for i in range(n):
        for j in range(i):
            sigma_n[i,j] = rho_fBM(i,j,h)
    return sigma_n + np.transpose(sigma_n) + np.eye(n)

def asymptotic_estimator(x_sample, H, T):
    c_H = gamma((H+1)/(2*H))*(2**(1/(2*H)))*np.pi**(-0.5)
    h_var_x = 0.0
    for i in range(len(x_sample)-1): 
        h_var_x = h_var_x + np.abs(x_sample[i+1] - x_sample[i])**(1/H)

    return ((1/(c_H*T))*h_var_x)**H

def mle_estimator(x_sample, h, T):
    x_sample = np.array(x_sample)
    n = x_sample.shape[0]
    x_increments = x_sample[1:] - x_sample[:n-1]
    delta = T/n
    cov_matrix_unnormalized = covariance_matrix_fmb(h, n-1)
    inverse_cov_unnormalized = np.linalg.inv(cov_matrix_unnormalized)
    return x_increments.dot(inverse_cov_unnormalized).dot(x_increments.T)/(n*(delta**(2*h)))

def x_epsilon(y_sample, epsilon, h, T):
    y_sample = np.array(y_sample)
    n = y_sample.shape[0]
    delta = T/n
    initial = 0
    realisation = [initial]
    for k in range(n): 
        initial += delta*(epsilon**(h-1))*y_sample[k]
        realisation.append(initial)
    return realisation 

def subsample(x_sample, epsilon, T, alpha):
    x_sample = np.array(x_sample)
    N = x_sample.shape[0]
    new_delta = epsilon**alpha 
    new_N = int(T/new_delta)
    if new_N > N: 
        raise Exception("Subsample size larger than original sample size")
    shift = int(N/new_N)

    x_subsample = [x_sample[0]]
    for i in range(1, int(N/shift)):
        x_subsample.append(x_sample[i*shift])
    return x_subsample

def covariance_x_epsilon(epsilon, delta, size):
    Q = np.zeros([size,size])
    for i in range(size): 
        for j in range(i):
            Q[i,j] = np.exp(-(i-j)/epsilon)
    
    Q = Q + np.transpose(Q) 
    P = ((epsilon*(np.exp(delta/epsilon)+np.exp(-delta/epsilon)-2))/2)*Q + np.eye(size)*(delta-epsilon*(1-np.exp(-delta/epsilon)))
    return P

def split_approx(x_sample, epsilon, T):
    x_sample = np.array(x_sample)
    n = x_sample.shape[0]
    x_increments = x_sample[1:] - x_sample[:n-1]

    delta = T/n
    P_eps = covariance_x_epsilon(epsilon, delta, n-1)

    P_eps_inv = np.linalg.inv(P_eps)
    M = (1/delta) * np.eye(n-1) - P_eps_inv
    reminder = (1/(n-1))*np.dot(x_increments, np.dot(M, x_increments))
    chi_term = (1/(n-1))*np.dot(x_increments, np.dot(P_eps_inv, x_increments))

    return [reminder, chi_term]

