import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import norm
from scipy.special import erfinv

def chance_constraint(beta, theta, sigma):
    lb = -np.sqrt(2)*sigma*erfinv(2* beta - 1)
    ub = -np.sqrt(2)*sigma*erfinv(1 - 2*theta)
    return lb, ub
lb, ub = chance_constraint(0.6, 0.6, 0.3)
print(ub)
# plt.close('all')
# mu = np.arange(100, -100, -1)
z = np.arange(0, 101, 1)
beta, theta = 0.9, 0.9
sigma = 10
lb, ub = chance_constraint(beta, theta, sigma)
mu1 = lb*np.ones(z.shape)
mu2 = ub*np.ones(z.shape)
x2 = np.zeros(z.shape)
fig, axs = plt.subplots(1,1)
axs.set_ylim([-100, 100])
axs.set_xlim([-100, 100])
axs.plot(x2, z, color = 'crimson', linewidth = 5)
axs.plot(z, x2, color = 'crimson', linewidth = 5)
width = 1
y1 = width * np.ones(z.shape)
y2 = -width * np.ones(z.shape)
# axs.fill_between(z, y1, y2, facecolor = 'crimson', label = 'Strict Feasible Region')
plt.xlabel('z')
plt.ylabel('$\mu_F$')
axs.fill_between(z, mu1, mu2, facecolor='lightgreen', label = 'Relaxed Feasible Region')
ticks = []
plt.xticks(ticks)
plt.yticks(ticks)
# fig.patch.set_visible(False)
axs.axis('off')
# axs.legend()
# plt.title('$\sigma$ = {f}'.format(f = sigma))
fig2, axs2 = plt.subplots(1,1)
axs2.set_ylim([-1, 30])
axs2.set_xlim([-1, 30])
axs2.plot(x2, z, color = 'crimson', linewidth = 3)
axs2.plot(z, x2, color = 'crimson', linewidth = 3)
y = np.zeros(z.shape)
y[0] = 200
y[1:] = 100/z[1:]
axs2.fill_between(z, y, x2, facecolor='xkcd:lavender', label = 'Relaxed Feasible Region')
axs2.axis('off')
plt.show()