import matplotlib.pyplot as plt
import numpy as np
def plot_ERM(horizontal_position, control, t, iteration, frictionVar):
    # Plot the horizontal trajectory
    fig1, axs1 = plt.subplots(2,1)
    for i in range(iteration):
        axs1[0].plot(t, horizontal_position[i,:], label = '$\sigma^2$ = {f}'.format(f = frictionVar[i]), linewidth=1.5)
        axs1[1].plot(t, control[i,:], linewidth=1.5)

    axs1[0].set_title('Horizontal Trajectory')
    axs1[0].set_ylabel('Position')
    axs1[0].legend()

    axs1[1].set_ylabel('Control (N)')
    axs1[1].set_xlabel('Time (s)')
    plt.show()

def plot_CC(horizontal_position, control, force, t, iteration, sigmas):
    # nominal trajectory
    x = np.loadtxt('data/slidingblock/warm_start/x.txt')
    u = np.loadtxt('data/slidingblock/warm_start/u.txt')
    l = np.loadtxt('data/slidingblock/warm_start/l.txt')
    t = np.loadtxt('data/slidingblock/warm_start/t.txt')
    
    # Plot the horizontal trajectory
    fig1, axs1 = plt.subplots(3,1)
    axs1[0].plot(t, x[0,:], 'k-', label = 'Reference', linewidth=1.5)
    axs1[1].plot(t, u[:],  'k-', linewidth=1.5)
    axs1[2].plot(t, l[1,:] - l[3,:], 'k-', linewidth=1.5)
    for i in range(iteration):
        axs1[0].plot(t, horizontal_position[i,:], label = '$\sigma$ = {f}'.format(f = sigmas[i]), linewidth=1.5)
        axs1[1].plot(t, control[i,:], linewidth=1.5)
        axs1[2].plot(t, force[i,:], linewidth=1.5)

    # axs1[0].set_title('Chance Constraint Variation w/ ERM w/ warmstart')
    axs1[0].set_title(' ERM w/ warmstart, no cc')
    axs1[0].set_ylabel('Position')
    axs1[0].legend()

    axs1[1].set_ylabel('Control (N)')
    axs1[2].set_ylabel('Friction (N)')
    axs1[2].set_xlabel('Time (s)')
    plt.show()

# x = np.loadtxt('data/slidingblock/warm_start/x.txt')
# u = np.loadtxt('data/slidingblock/warm_start/u.txt')
# l = np.loadtxt('data/slidingblock/warm_start/l.txt')
# t = np.loadtxt('data/slidingblock/warm_start/t.txt')
# # # Plot the horizontal trajectory
# fig1, axs1 = plt.subplots(3,1)
# axs1[0].plot(t, x[0,:], 'k-', label = 'Reference', linewidth=1.5)
# axs1[1].plot(t, u[:],  'k-', linewidth=1.5)
# axs1[2].plot(t, l[1,:] - l[3,:], 'k-', linewidth=1.5)
# plt.show()