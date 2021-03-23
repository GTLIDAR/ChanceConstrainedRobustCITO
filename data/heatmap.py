import numpy as np
import matplotlib.pyplot as plt
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
import matplotlib.colors as colors
from scipy.stats import norm
def ermCost(x, mu, sigma):
    """
    Gaussian ERM implementation
    """
    # initialize pdf and cdf
    pdf = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(x, mu, sigma)
    f = x**2 - sigma**2 * (x + mu) * pdf + (sigma**2+ mu**2 - x**2) * cdf
    return f

mu = np.arange(100, -100, -1)
z = np.arange(-100, 100, 1)
sigma = 10
n = len(mu)
erm = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        erm[i][j] = ermCost(z[j].reshape(1,), mu[i].reshape(1,), sigma)
        if erm[i][j] < 1e-4:
            # print(i, j)
            erm[i][j] = 1e-5


fig, ax = plt.subplots(1,1)

norm = colors.LogNorm(vmin = 0.001, vmax = np.max(erm))
# pcm = ax.pcolor(mu, z, erm, norm = norm, cmap='magma')
# fig.colorbar(pcm, ax = ax, extend = 'max')
plt.imshow(erm, cmap= 'Blues', extent =[-100, 100, -100, 100], norm = norm)
ticks = [-100, -50, 0, 50, 100]
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel('z')
plt.ylabel('$\mu_F$')
plt.title('$\sigma$ = {f}'.format(f = sigma))
plt.colorbar()
plt.show()
print('Done!')


