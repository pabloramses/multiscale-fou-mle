import numpy as np 
import matplotlib.pyplot as plt

#LaTeX
A=6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#Data
results = np.load("results08.npy")
epsilon = np.linspace(0.001, 0.1, 10)

results = np.flip(results, axis=0)
# Plotting
cmap = plt.cm.get_cmap('Greens')

# Plotting each row of results with gradient colors
for i, row in enumerate(results):
    color = cmap(i / (len(results) - 1))  # Adjust color based on row index
    plt.plot(epsilon, row, label=fr'$\alpha=$ {round((i+1)*0.1,1)}', color=color)

plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\overline{\tilde{\sigma}}^{10}$')
plt.title(r'Mean $\tilde{\sigma}$ as $\epsilon\rightarrow 0$ with $\delta=\epsilon^{\alpha}$ and \textbf{H=0.8}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('H_08.png',bbox_inches='tight')
plt.tight_layout()
plt.show()
