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
        # xtraj, utraj, ftraj, _ = trajopt.reconstruct_all_trajectories(result)
        # [t, u] = GetKnotsFromTrajectory(soln['control'])
        axs1[1].plot(t, u, label=f'$\sigma$ ={sigma}', linewidth=2.5)
    axs1[1].legend()
    plt.show()
    
if __name__ == "__main__":
    plot_control_trajectories(folder="data/IEEE_Access/sliding_block/ERM", name = "block_erm",
                    sigmas=np.array([0.01, 0.05, 0.1, 0.3, 1]))
    pass