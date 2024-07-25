import numpy as np 
import matplotlib.pyplot as plt 
from stochastic.processes.continuous import FractionalBrownianMotion
from fOU import fOU

fbm_05 = FractionalBrownianMotion(hurst = 0.5, t = 10)
fbm_025 = FractionalBrownianMotion(hurst = 0.25, t = 10)
fbm_075 = FractionalBrownianMotion(hurst = 0.75, t = 10)
time = np.linspace(0,10,1001)

plt.plot(time,fbm_025.sample(1000))
plt.title(r'fBM with $H=1/4$')
plt.savefig('fbm_025',bbox_inches='tight')
plt.close()
plt.plot(time,fbm_05.sample(1000))
plt.title(r'fBM with $H=1/2$')
plt.savefig('fbm_05',bbox_inches='tight')
plt.close()
plt.plot(time,fbm_075.sample(1000))
plt.title(r'fBM with $H=3/4$')
plt.savefig('fbm_075',bbox_inches='tight')
plt.close()

fou_025 = fOU(0.25, 10, 2, 1)
fou_05 = fOU(0.5, 10, 2, 1)
fou_075 = fOU(0.75, 10, 2, 1)
plt.plot(time,fou_025.sample(1000))
plt.title(r'fOU with $H=1/4$')
plt.savefig('fou_025',bbox_inches='tight')
plt.close()
plt.plot(time,fou_05.sample(1000))
plt.title(r'fOU with $H=1/2$')
plt.savefig('fou_05',bbox_inches='tight')
plt.close()
plt.plot(time,fou_075.sample(1000))
plt.title(r'fOU with $H=3/4$')
plt.savefig('fou_075',bbox_inches='tight')
plt.close()
