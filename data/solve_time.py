import numpy as np
import matplotlib.pyplot as plt
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
import matplotlib.colors as colors
# import pandas as pd
# sigma = 0.05
time1 = np.array([[219.2, 91.3, 59.7], [332.5, 85.8, 119.0], [142.7, 312.9, 89.9]])
# sigma = 0.1
time2 = np.array([[91.0, 32.0, 178.4], [168.9, 35.6, 27.0], [33.3, 173.6, 107.4]])
# sigma = 0.3
time3 = np.array([[6.6, 92.2, 84.6], [109.5, 228.9, 65.8], [41.3, 37.6, 60.4]])
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
