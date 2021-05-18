from systems.timestepping import TimeSteppingMultibodyPlant
import numpy as np
import matplotlib.pyplot as plt
from systems.terrain import FlatTerrain

_file = "systems/urdf/sliding_block.urdf"
cc_erm_control = np.loadtxt('data/slidingblock/erm_w_cc/control.txt')
erm_control= np.loadtxt('data/slidingblock/erm/control.txt')
cc_erm_control = cc_erm_control.reshape([5, 101])
erm_control = erm_control.reshape([5, 101])
reference_control = np.loadtxt('data/slidingblock/warm_start/u.txt')
# u = u.reshape([1, 101])
h = 0.01
frictions = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
x0 = np.array([0, 0.5, 0, 0])

reference = np.ones([101, 1]) * 5
reference_final_position = np.zeros(frictions.shape)
erm_final_position = np.zeros([5,len(frictions)])
cc_erm_final_position = np.zeros([5,len(frictions)])
final_position = np.zeros([11, len(frictions)])
# reference control
for i in range(len(frictions)):
    plant = TimeSteppingMultibodyPlant(file= _file, terrain = FlatTerrain(friction = frictions[i]))
    plant.Finalize()
    t, x_ref, f = plant.simulate(h, x0, u = reference_control.reshape(1,101), N = 101)
    final_position[0, i] = x_ref[0, 100]
    for n in range(5): 
        t, x_erm, f = plant.simulate(h, x0, u = erm_control[n, :].reshape(1,101), N = 101)
        t, x_erm_cc, f = plant.simulate(h, x0, u = cc_erm_control[n, :].reshape(1,101), N = 101)
        final_position[2*n + 1,i] = x_erm[0,100]
        final_position[2*n + 2,i] = x_erm_cc[0,100]
    # axs1.plot(t, x[0,:], linewidth=1.5, label = '$\mu$ = {f}'.format(f = frictions[i]))
    # axs1.set_title('ERM + CC Control $\sigma$ = 1')
    # axs1.set_ylabel('Position')
fig, ax= plt.subplots(1,1)
# x = np.arange(0,11,1)
cc = np.array([False, False, True, False, True, False, True, False, True, False, True])
erm = np.array([False, True, False, True, False, True, False, True, False, True, False])
ref = np.array([True, False, False, False, False, False, False, False, False, False, False])

x = np.arange(0,len(cc),1)
error = final_position - 5
y = np.mean(error, axis = 1)

lower_error = np.abs(error.min(axis = 1) - y)
upper_error = np.abs(error.max(axis = 1) - y)
# print(lower_error)
# print(upper_error)
print(lower_error + upper_error)
asym_error = np.array([lower_error, upper_error])
xref = np.array([0])
ax.errorbar(xref,y[ref], yerr = [lower_error[ref], upper_error[ref]], fmt = 'o', capsize = 4, label = 'Reference',markersize='5', color = 'darkorange', elinewidth = 2, capthick = 2)
xerm = np.array([3, 7, 11, 15, 19])
# ax.errorbar(x[not cc],y[not cc], yerr = asym_error[not cc, :], fmt = 'ok', capsize = 3)
ax.errorbar(xerm,y[erm], yerr = [lower_error[erm], upper_error[erm]], fmt = 'o', capsize = 4, label = 'ERM',markersize='5', color = 'darkblue', elinewidth = 2, capthick = 2)
xcc = np.array([4, 8, 12, 16, 20])
ax.errorbar(xcc,y[cc], yerr = [lower_error[cc], upper_error[cc]], fmt = 'o', capsize = 4, label = 'CC + ERM',markersize='5', color = 'darkgreen', elinewidth = 2, capthick = 2)
x_bar = np.arange(-1, 21, 1)
y_bar = np.zeros(x_bar.shape)
ax.set_xlim([-1, 21])
ax.plot(x_bar, y_bar, 'k--')
ticks = ['Reference', '$\sigma = 0.01$', '$\sigma = 0.01$', '$\sigma = 0.05$', '$\sigma = 0.05$', '$\sigma = 0.1$','$\sigma = 0.1$', '$\sigma = 0.3$', '$\sigma = 0.3$', '$\sigma = 1$', '$\sigma = 1$']
ticks = []
ax.set_xticks([])
# ax.set_xticklabels(ticks)
plt.xticks(rotation = 30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Position Error')
# axs1.plot(t, reference, 'k-', linewidth = 1.5, label = 'Target' )    
# ax.legend() 
plt.show()