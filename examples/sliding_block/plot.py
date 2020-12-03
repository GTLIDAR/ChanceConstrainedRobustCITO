import numpy as np
import matplotlib.pyplot as plt

x_strict = np.loadtxt('x_strict.txt', dtype = float)
u_strict = np.loadtxt('u_strict.txt', dtype = float)
x_relaxed = np.loadtxt('x_relaxed.txt', dtype = float)
u_relaxed = np.loadtxt('u_relaxed.txt', dtype = float)
# l_strict = np.loadtxt('l_strict.txt', dtype = int)
t = np.loadtxt('time.txt', dtype = float)
params = {'legend.fontsize': 40}
fig, axs = plt.subplots(2,1)
relaxed, = axs[0].plot(t, x_relaxed[0,:], 'b-', label = "relaxed")
strict, = axs[0].plot(t, x_strict[0,:], 'r-', label = "strict")

axs[0].set_ylabel('Position')
axs[0].legend(handles = [relaxed, strict], loc = 'upper left', prop = {'size': 15})
axs[0].tick_params(axis='x', which='both',length=0, labelsize = 0)
axs[0].tick_params(axis='y', which='both',length=0, labelsize = 0)

strict1, = axs[1].plot(t, u_strict[:], 'r-', label = "strict")
relaxed1, = axs[1].plot(t, u_relaxed[:], 'b-', label = "relaxed")
# axs[1].legend(handles = [strict, relaxed], loc = 'upper right')
axs[1].tick_params(axis='x', which='both',length=0, labelsize = 0)
axs[1].tick_params(axis='y', which='both',length=0, labelsize = 0)
axs[1].set_ylabel('Control')
axs[1].set_xlabel('Time')


# Show the plots
plt.show()