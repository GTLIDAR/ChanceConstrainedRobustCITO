"""This is used for generateting visualization for the IEEE Access paper"""
import numpy as np
import matplotlib.pyplot as plt
from utilities import FindResource, GetKnotsFromTrajectory, load

def plot_control_trajectories(folder = None, name = None, sigmas = None):
    fig1, axs1 = plt.subplots(3,1)
    ref_soln = load('data/IEEE_Access/sliding_block/block_trajopt_nominal.pkl')
    t = ref_soln['time'].reshape(101,1)
    ref_u = ref_soln['control'].reshape(101,1)
    axs1[0].plot(t, ref_u, label='Reference', linewidth=2.5)
    axs1[0].legend()
    for sigma in sigmas:
        filename = f"{folder}/{name}_{sigma}"
        soln = load(filename)
        u = soln['control'].reshape(101,1)
        f = soln['force']
        # xtraj, utraj, ftraj, _ = trajopt.reconstruct_all_trajectories(result)
        # [t, u] = GetKnotsFromTrajectory(soln['control'])
        axs1[1].plot(t[:-1], u[:-1], label=f'$\sigma$ ={sigma}', linewidth=2.5)
        axs1[2].plot(t[:-1], f[1,:-1]-f[3,:-1], label=f'$\sigma$ ={sigma}', linewidth=2.5)
    axs1[1].legend()
    plt.show()

def compare_traj():
    x_old= np.loadtxt('data/slidingblock/warm_start/x.txt')
    u_old = np.loadtxt('data/slidingblock/warm_start/u.txt')
    # u_init = u_init.reshape(trajopt.u.shape)
    l_old = np.loadtxt('data/slidingblock/warm_start/l.txt')
    soln = load("data/IEEE_Access/sliding_block/block_trajopt_nominal_tight.pkl")
    u_new = soln['control'].reshape(101,1)
    x_new = soln['state']
    l_new = soln['force']
    t = soln['time'].reshape(101,1)
    fig1, axs1 = plt.subplots(3,1)
    axs1[0].plot(t, u_old, label='old', linewidth=2.5)
    axs1[1].plot(t, x_old[0], label='old', linewidth=2.5)
    axs1[2].plot(t, l_old[1,:]-l_old[3,:], label='old', linewidth=2.5)
    fig2, axs2 = plt.subplots(3,1)
    axs2[0].plot(t, u_new, label='new', linewidth=2.5)
    axs2[1].plot(t, x_new[0], label='new', linewidth=2.5)
    axs2[2].plot(t, l_new[1,:]-l_new[3,:], label='new', linewidth=2.5)
    plt.show()

if __name__ == "__main__":
    plot_control_trajectories(folder="data/IEEE_Access/sliding_block/ERM_tight", name = "block_erm",
                    sigmas=np.array([0.01, 0.05, 0.1, 0.3, 1]))
                    
    # compare_traj()
    pass