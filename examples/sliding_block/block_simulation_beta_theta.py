from systems.timestepping import TimeSteppingMultibodyPlant
import numpy as np
import matplotlib.pyplot as plt
from systems.terrain import FlatTerrain

_file = "systems/urdf/sliding_block.urdf"

cc_erm_control= np.loadtxt('data/slidingblock/beta_theta_0.1/control.txt')
cc_erm_control = cc_erm_control.reshape([9, 101])

reference_control = np.loadtxt('data/slidingblock/warm_start/u.txt')
erm_control = np.loadtxt('data/slidingblock/erm/control.txt')
erm_control = erm_control.reshape([5, 101])
erm_control = erm_control[3,:]
# u = u.reshape([1, 101])
h = 0.01
frictions = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
x0 = np.array([0, 0.5, 0, 0])

reference = np.ones([101, 1]) * 5
reference_final_position = np.zeros(frictions.shape)
erm_final_position = np.zeros([9,len(frictions)])
cc_erm_final_position = np.zeros([9,len(frictions)])
final_position = np.zeros([10, len(frictions)])
# reference control
for i in range(len(frictions)):
    plant = TimeSteppingMultibodyPlant(file= _file, terrain = FlatTerrain(friction = frictions[i]))
    plant.Finalize()
    # t, x_ref, f = plant.simulate(h, x0, u = reference_control.reshape(1,101), N = 101)
    # final_position[0, i] = x_ref[0, 100]
    t, x_erm, f = plant.simulate(h, x0, u = erm_control.reshape(1,101), N = 101)
    final_position[0, i] = x_erm[0, 100]

    for n in range(9): 
        t, x_erm_cc, f = plant.simulate(h, x0, u = cc_erm_control[n, :].reshape(1,101), N = 101)
        final_position[n + 1,i] = x_erm_cc[0,100]
    # axs1.plot(t, x[0,:], linewidth=1.5, label = '$\mu$ = {f}'.format(f = frictions[i]))
    # axs1.set_title('ERM + CC Control $\sigma$ = 1')
    # axs1.set_ylabel('Position')
fig, ax= plt.subplots(1,1)
x = np.arange(0,10,1)
cc_ref = np.array([True, False, False,False,False,False,False,False,False,False])
cc1 = np.array([False, True, True, True, False, False, False, False, False, False])
cc2 = np.array( [False, False, False, False, True, True, True, False, False, False])
cc3 = np.array([False, False, False, False, False, False, False, True, True, True])
x1 = np.array([False, True, False, False, True, False, False, True, False, False])
x2 = np.array([False, False, True, False, False, True, False,  False, True, False])
x3 = np.array([False, False, False, True,False, False, True, False, False, True])
error = final_position - 5
y = np.mean(error, axis = 1)
lower_error = np.abs(error.min(axis = 1) - y)
upper_error = np.abs(error.max(axis = 1) - y)
asym_error = np.array([lower_error, upper_error])
print(y)
print(lower_error + upper_error)
x_bar = np.arange(-1, 19, 1)
y_bar = np.zeros(x_bar.shape)
ax.plot(x_bar, y_bar, 'k--')
ax.set_xlim([-1,19])
ax.set_yticks([0.4, 0.2, 0, -0.2])
# ax.set_ylim([-0.21, 0.41])
# ax.errorbar(x[not cc],y[not cc], yerr = asym_error[not cc, :], fmt = 'ok', capsize = 3)
ax.errorbar(x[cc_ref],y[cc_ref], yerr = [lower_error[cc_ref], upper_error[cc_ref]], fmt = 'o', capsize = 4,markersize='5', color = 'darkblue', label = 'ERM', elinewidth = 2, capthick = 2)
x1 = np.array([3, 9, 15])
x2 = np.array([4, 10, 16])
x3 = np.array([5, 11, 17])
ax.errorbar(x1,y[cc1], yerr = [lower_error[cc1], upper_error[cc1]], fmt = 'o', capsize = 4,markersize='5', color = 'tab:green', label = 1, elinewidth = 2, capthick = 2)
ax.errorbar(x2,y[cc2], yerr = [lower_error[cc2], upper_error[cc2]], fmt = 'o', capsize = 4,markersize='5', color = 'tab:purple', label = 2, elinewidth = 2, capthick = 2)
ax.errorbar(x3,y[cc3], yerr = [lower_error[cc3], upper_error[cc3]], fmt = 'o', capsize = 4,markersize='5', color = 'tab:gray', label = 3, elinewidth = 2, capthick = 2)
ticks = ['ERM', '$\\theta = 0.51$', '$\\theta = 0.6$', '$\\theta = 0.9$']
ax.set_xticks([])
ax.set_xticklabels(ticks)
plt.xticks(rotation = 30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend()
# ax[0].set_ylabel('$\\beta = 0.51$')
# ax[1].set_ylabel('$\\beta = 0.6$')
# ax[2].set_ylabel('$\\beta = 0.9$')
# ax[0].set_title('$\sigma = 0.3$')
# ax[0].set_ylim([-0.5, 0.5])
# ax[1].set_ylim([-0.5, 0.5])
# ax[2].set_ylim([-0.5, 0.5])
# ax[0].set_yticks([-0.5, 0, 0.5])
# ax[1].set_yticks([-0.5, 0, 0.5])
# ax[2].set_yticks([-0.5, 0, 0.5])
# ax[0].set_xticks([])
# ax[1].set_xticks([])
# axs1.plot(t, reference, 'k-', linewidth = 1.5, label = 'Target' )    
# ax.legend() 
plt.show()