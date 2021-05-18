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

def plot_CC(horizontal_position, control, force, t, sigmas):
    # nominal trajectory
    x = np.loadtxt('data/slidingblock/warm_start/x.txt')
    u = np.loadtxt('data/slidingblock/warm_start/u.txt')
    l = np.loadtxt('data/slidingblock/warm_start/l.txt')
    t = np.loadtxt('data/slidingblock/warm_start/t.txt')
    iteration = len(sigmas  )
    # Plot the horizontal trajectory
    fig1, axs1 = plt.subplots(3,1)
    
    for i in range(iteration):
        axs1[0].plot(t, horizontal_position[i,:], label = '$\sigma$ = {f}'.format(f = sigmas[i]), linewidth=1.5)
        axs1[1].plot(t[:-1], control[i,:-1], linewidth = 2.5)
        axs1[2].plot(t[1:-1], force[i,1:-1], linewidth= 2.5)
    ref_x = axs1[0].twinx()
    ref_u = axs1[1].twinx()
    ref_u.set_ylabel('Reference')
    ref_f = axs1[2].twinx()
    # plot reference traj
    # axs1[0].plot(t, x[0,:], 'k-', label = 'Reference', linewidth=1.5)
    # axs1[1].plot(t, u[:],  'k-', linewidth=1.5)
    # axs1[2].plot(t, l[1,:] - l[3,:], 'k-', linewidth=1.5)
    ref_x.plot(t, x[0,:], 'k-', linewidth= 2.5)
    ref_u.plot(t[:-1], u[:-1],  'k-', linewidth=2.5)
    ref_f.plot(t[1:], l[1,1:] - l[3,1:], 'k-', linewidth=2.5)
    ref_f.set_yticks([0,-2, -4, -6, -8])
    ref_f.set_ylabel('Reference')
    ref_x.set_ylabel('Reference')
    axs1[2].set_yticks([0,-2, -4, -6, -8])
    # ref_x.legend()
    # axs1[0].set_title('Chance Constraint Variation w/ ERM w/ warmstart')
    axs1[0].set_title('ERM')
    axs1[0].set_ylabel('Position')
    axs1[0].set_xlim([0,1])
    axs1[0].legend()
    # axs1[1].set_yticks([500, 250, 0, -250, -500])
    axs1[1].set_yticks([-50, 0, 50])
    ref_u.set_yticks([-50, 0, 50])
    ref_f.set_yticks([0,-2, -4, -6, -8])
    axs1[1].set_ylabel(' ERM Control (N)')
    axs1[1].set_xlim([0,1])
    axs1[2].set_ylabel('ERM Friction (N)')
    axs1[2].set_xlim([0,1])
    axs1[2].set_ylim([ -8, 0.5])
    ref_f.set_ylim([ -8, 0.5])
    # axs1[2].set_ylim([-6, 1])
    # ref_f.set_ylim([-6,1])
    ref_u.set_ylim([-50, 50])
    axs1[0].spines['right'].set_visible(False)
    axs1[1].spines['top'].set_visible(False)
    axs1[1].spines['bottom'].set_visible(False)
    axs1[1].spines['left'].set_visible(False)
    axs1[2].spines['right'].set_visible(False)
    axs1[2].spines['top'].set_visible(False)
    axs1[2].spines['bottom'].set_visible(False)
    axs1[2].spines['left'].set_visible(False)
    axs1[2].set_xlabel('Time (s)')
    ref_u.spines['right'].set_visible(False)
    plt.show()

def plot_CC_beta_theta(horizontal_position, control, force, t, beta_theta):
    # nominal trajectory
    x = np.loadtxt('data/slidingblock/warm_start/x.txt')
    u = np.loadtxt('data/slidingblock/warm_start/u.txt')
    l = np.loadtxt('data/slidingblock/warm_start/l.txt')
    t = np.loadtxt('data/slidingblock/warm_start/t.txt')
    iteration = len(beta_theta)
    # Plot the horizontal trajectory
    fig1, axs1 = plt.subplots(3,1)
    
    for i in range(iteration):
        axs1[0].plot(t, horizontal_position[i,:], label = '$\\beta$ = {b}, $\\theta = {t}$'.format(b = beta_theta[i,0], t = beta_theta[i, 1]), linewidth=1.5)
        axs1[1].plot(t, control[i,:], linewidth=1.5)
        axs1[2].plot(t, force[i,:], linewidth=1.5)
    ref_x = axs1[0].twinx()
    ref_u = axs1[1].twinx()
    ref_u.set_ylabel('Reference Control (N)')
    ref_f = axs1[2].twinx()
    # plot reference traj
    # axs1[0].plot(t, x[0,:], 'k-', label = 'Reference', linewidth=1.5)
    # axs1[1].plot(t, u[:],  'k-', linewidth=1.5)
    # axs1[2].plot(t, l[1,:] - l[3,:], 'k-', linewidth=1.5)
    ref_x.plot(t, x[0,:], 'k-', label = 'Reference', linewidth=1.5)
    ref_u.plot(t[:-1], u[:-1],  'k-', linewidth=1.5)
    ref_f.plot(t, l[1,:] - l[3,:], 'k-', linewidth=1.5)
    ref_f.set_yticks([0,-2, -4, -6, -8])
    ref_f.set_ylabel('Reference Friction (N)')
    ref_x.set_ylabel('Reference position (m)')
    axs1[2].set_yticks([0,-2, -4, -6, -8])
    ref_x.legend()
    # axs1[0].set_title('Chance Constraint Variation w/ ERM w/ warmstart')
    axs1[0].set_title('ERM')
    axs1[0].set_ylabel('Position')
    axs1[0].set_xlim([0,1])
    axs1[0].legend()

    axs1[1].set_ylabel('Control (N)')
    
    axs1[1].set_xlim([0,1])
    axs1[2].set_ylabel('Friction (N)')
    axs1[2].set_xlim([0,1])
    axs1[2].set_ylim([-8, 1])
    axs1[2].set_xlabel('Time (s)')
    plt.show()

if __name__ == "__main__":
    horizontal_position = np.loadtxt('data/slidingblock/erm/horizontal_position.txt')
    control = np.loadtxt('data/slidingblock/erm/control.txt')
    friction = np.loadtxt('data/slidingblock/erm/friction.txt')
    t = np.loadtxt('data/slidingblock/erm/t.txt')
    sigmas = [0.01,0.05, 0.1, 0.3, 1]
    params = np.array([0.51, 0.6, 0.9])
    # beta_theta = np.array([[params[0], params[0]], [params[0], params[1]], [params[0], params[2]],
    #                     [params[1], params[0]], [params[1], params[1]], [params[1], params[2]],
    #                     [params[2], params[0]], [params[2], params[1]], [params[2], params[2]]])
    # plot_CC_beta_theta(horizontal_position, control, friction, t, beta_theta)

    plot_CC(horizontal_position, control, friction, t, sigmas)