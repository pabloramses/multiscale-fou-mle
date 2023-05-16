import numpy as np 
import matplotlib.pyplot as plt 
from utils import ensure_single_arg_constant_function
from stochastic.processes.continuous import FractionalBrownianMotion

"""
Simulation from a fracional Ornstein-Uhlenbeck process that follows the SDE 

dYt = -1/epsilon Yt*dt + sigma/epsilon^H dfBMtH 

where fBMtH is an standard fractional Brownian Motion with hurst parameter H
"""

class fOU:
    def __init__(self, H, T, sigma = 1, epsilon = 0.1):
        self.H = H
        self.T = T 
        self.fbm = self.fractional_noise
        self.sigma = sigma 
        self.scale = epsilon 

    @property
    def fractional_noise(self):
        return FractionalBrownianMotion(hurst = self.H, t = self.T)
    
    @property
    def scale(self): 
        return self._scale 
    
    @scale.setter 
    def scale(self, value): 
        self._scale = ensure_single_arg_constant_function(value) 

    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        self._sigma = ensure_single_arg_constant_function(value)


    def sample(self, n, initial = 0.0):
        """
        Employs a first order Euler-Maruyama scheme with the precaution that the traslation of our notation to conventional is:
        speed = 1/scale
        mean = 0
        volatility = sigma/(epsilon^H)
        exponential volatility = 0
        """
        delta = 1.0 * self.T / n
        f_BM = self.fractional_noise.sample(n+1)

        realisation = [initial]
        t = 0
        
        for k in range(n):
            t += delta 
            initial += ( 
                 -((1/self._scale(t)) * initial) * delta + (self._sigma(t)/(self._scale(t)**self.H))*(f_BM[k+1]- f_BM[k])  )
            realisation.append(initial)
        return realisation

