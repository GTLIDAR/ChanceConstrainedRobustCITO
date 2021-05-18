import numpy as np
import matplotlib.pyplot as plt
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
import matplotlib.colors as colors
# import pandas as pd
# sigma = 0.05
time1 = np.array([[12.4, 71.1, 204.1], [211.2, 86.6, 286.0], [258.0, 291.6, 223.4]])
# sigma = 0.1
time2 = np.array([[36.5, 169.5, 133.6], [177.5, 152.4, 135.6], [152.3, 166.4, 192.6]])
# sigma = 0.3
time3 = np.array([[21.8, 50.9, 78.8], [42.4, 49.9, 73.7], [77.5, 68.7, 73.8]])
# sigma = 0.5
time4 = np.array([[35.2, 37.4, 53.0], [45.0, 55.6, 65.4], [53.9, 31.7, 56.0]]) 
c = 'Blues'
fig, ax = plt.subplots(1,3)
norm = colors.LogNorm(vmin = 1, vmax = 300)
im = ax[0].imshow(time1, cmap= c, vmin = 0, vmax = 300)
ax[0].set_title('$\sigma = 0.05$')
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticks([])
ax[2].set_yticks([])
ax[2].set_xticks([])
ax[0].set_ylabel('$\\beta$')
ax[0].set_xlabel('$\\theta$')
ax[1].set_xlabel('$\\theta$')
ax[2].set_xlabel('$\\theta$')
ax[1].set_title('$\sigma = 0.1$')
ax[2].set_title('$\sigma = 0.3$')
# ax[3].set_title('$\sigma = 0.05$')
ax[1].imshow(time2, cmap= c, vmin = 0, vmax = 300)
ax[2].imshow(time3, cmap= c, vmin = 0, vmax = 300)
# im = ax[3].imshow(time4, cmap= 'Blues', vmin = 0, vmax = 250)
cbar = fig.colorbar(im)

plt.show()
