import numpy as np
import matplotlib.pyplot as plt
from systems.terrain import FlatTerrain
from systems.timestepping import TimeSteppingMultibodyPlant
_file = "systems/urdf/sliding_block.urdf"
cc_erm_control = np.loadtxt('data/slidingblock/erm_w_cc/control.txt')
erm_control= np.loadtxt('data/slidingblock/erm/control.txt')
cc_erm_control = cc_erm_control.reshape([5, 101])
erm_control = erm_control.reshape([5, 101])
reference_control = np.loadtxt('data/slidingblock/warm_start/u.txt')
# u = u.reshape([1, 101])
h = 0.01
frictions = np.array([ 0.3, 0.43, 0.57, 0.7])
x0 = np.array([0, 0.5, 0, 0])
fig, axs = plt.subplots(3,1)
for i in range(len(frictions)):
    plant = TimeSteppingMultibodyPlant(file= _file, terrain = FlatTerrain(friction = frictions[i]))
    plant.Finalize()
    control = reference_control
    t, x, f = plant.simulate(h, x0, u = control.reshape(1,101), N = 101)
    y_bar = np.zeros(t.shape) + 5
    axs[0].plot(t, y_bar, 'k', linewidth =1)
    axs[0].plot(t, x[0, :], linewidth =3, label = '$\mu$ = {f}'.format(f = frictions[i]))
    axs[0].set_yticks([0, 2, 4, 6])
    axs[0].set_ylim([0,6.1])
    axs[0].set_xlim([0,1])
    # Hide the right and top spines
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].yaxis.set_ticks_position('left')

for i in range(len(frictions)):
    plant = TimeSteppingMultibodyPlant(file= _file, terrain = FlatTerrain(friction = frictions[i]))
    plant.Finalize()
    control = erm_control[4]
    t, x, f = plant.simulate(h, x0, u = control.reshape(1,101), N = 101)
    y_bar = np.zeros(t.shape) + 5
    axs[1].plot(t, y_bar, 'k', linewidth =1)
    axs[1].plot(t, x[0, :], linewidth =3, label = '$\mu$ = {f}'.format(f = frictions[i]))
    axs[1].set_yticks([0, 2, 4, 6])
    axs[1].set_ylim([-0.1,6.1])
    axs[1].set_xlim([0,1])
    # Hide the right and top spines
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

for i in range(len(frictions)):
    plant = TimeSteppingMultibodyPlant(file= _file, terrain = FlatTerrain(friction = frictions[i]))
    plant.Finalize()
    control = cc_erm_control[4]
    t, x, f = plant.simulate(h, x0, u = control.reshape(1,101), N = 101)
    y_bar = np.zeros(t.shape) + 5
    axs[2].plot(t, y_bar, 'k', linewidth =1)
    axs[2].plot(t, x[0, :], linewidth =3, label = '$\mu$ = {f}'.format(f = frictions[i]))
    axs[2].set_yticks([0, 2, 4, 6])
    axs[2].set_ylim([-0.1,6.1])
    axs[2].set_xlim([0,1])
    # Hide the right and top spines
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['bottom'].set_visible(False)
plt.show()