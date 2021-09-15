import numpy as np
import decorators as deco
import matplotlib.pyplot as plt
import utilities as utils
import os
from pydrake.all import PiecewisePolynomial
from systems.hopper.hopper import Hopper
import decorators as deco

def get_hopper_distance_trajectory(hopper, state):
    context = hopper.multibody.CreateDefaultContext()
    distance = np.zeros((2, state.shape[1]))
    for n in range(state.shape[1]):
        hopper.multibody.SetPositionsAndVelocities(context, state[:,n])
        distance[:,n] = hopper.GetNormalDistances(context)
    return distance

@deco.showable_fig
@deco.saveable_fig
def plot_foot_distances(hopper, state, forces):
    t, x = utils.GetKnotsFromTrajectory(state)
    _, f = utils.GetKnotsFromTrajectory(forces)
    # Get distances
    d = get_hopper_distance_trajectory(hopper, x)
    fig, axs = plt.subplots(2,1)
    utils.plot_complementarity(axs[0], t, d[0,:], f[0,:], 'Distance','Force')
    utils.plot_complementarity(axs[1], t, d[1,:], f[1,:], 'Distance', 'Force')
    return fig, axs

def main():
    directory = os.path.join('examples','hopper','robust_erm_hotfix_1e6_linear')
    file = 'trajoptresults.pkl'
    hopper = Hopper()
    hopper.Finalize()
    for pathname in utils.find_filepath_recursive(directory, file):
        print(f"Working on file {os.path.join(pathname, file)}")
        data = utils.load(os.path.join(pathname, file))
        xtraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])
        ftraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['force'])
        plot_foot_distances(hopper, xtraj, ftraj, show=False, savename=os.path.join(pathname, 'DistanceComplementarity.png'))

if __name__ == "__main__":
    main()