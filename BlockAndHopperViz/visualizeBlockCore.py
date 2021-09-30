import numpy as np
from utilities import load, FindResource  
from systems.visualization import Visualizer
from pydrake.all import RigidTransform, PiecewisePolynomial, Rgba, RoleAssign

# Colors - the matlab default colors, for comparison, plus some greys
colors = {'blue': np.array([0.0000, 0.4470, 0.7410, 1.]),
        'orange': np.array([0.8500, 0.3250, 0.0980, 1.]),
        'yellow': np.array([0.9290, 0.6940, 0.1250, 1.]),
        'purple': np.array([0.4940, 0.1840, 0.5560, 1.]),
        'green':  np.array([0.4660, 0.6740, 0.1880, 1.]),
        'cyan':   np.array([0.3010, 0.7450, 0.9330, 1.]),
        'red':    np.array([0.6350, 0.0780, 0.1840, 1.]),
        'black':  np.array([0., 0., 0., 1.]),
        'white':  np.array([1., 1., 1., 1.]),
        'grey':   np.array([0.8, 0.8, 0.8, 1.]),
        'lightgrey': np.array([0.4, 0.4, 0.4, 1.]),
        'medgrey': np.array([0.5, 0.5, 0.5, 1.]),
        'brown': np.array([0.549, 0.337, 0.294, 1.0]),
        'pink': np.array([0.890, 0.467, 0.761, 1.0])}

#Helper function
def weld_base_to_world(vis, model_index, translation=np.zeros((3,))):
    """removes floating base variables by welding base to world"""
    body_ind = vis.plant.GetBodyIndices(model_index)
    body_frame = vis.plant.get_body(body_ind[0]).body_frame()
    vis.plant.WeldFrames(vis.plant.world_frame(), body_frame, RigidTransform(translation))
    return vis

def set_block_color(vis, model_index, color=np.array([1, 1, 1, 1])):
    """Changes the color of the block in the visualizer"""
    # Get the frame ID for block
    body_inds = vis.plant.GetBodyIndices(model_index)
    vis.setBodyColor(body_inds[-1], color)
    return vis

def add_terrain(visualizer, terrain_color=None):
    # Add in the terrain and weld
    visualizer.addModelFromFile(FindResource("systems/urdf/blockterrain.urdf"))
    visualizer = weld_base_to_world(visualizer, visualizer.model_index[-1])
    if terrain_color is not None:
        terrain_inds = visualizer.plant.GetBodyIndices(visualizer.model_index[-1])
        for n in [0, 2]:
           visualizer.setBodyColor(terrain_inds[n], terrain_color)
    return visualizer

def visualize_multiblock(datasets, colors, terraincolor=None, key="state"):
    """ Visualize multiple blocks at once """
    nBlocks = len(datasets)
    # Create a block visualizer with multiple blocks
    blockfile = "systems/urdf/sliding_block.urdf"
    visualizer = Visualizer(FindResource(blockfile))
    # visualizer = Visualizer(blockfile)
    for n in range(1,nBlocks):
        visualizer.addModelFromFile(FindResource(blockfile), name="Block"+str(n))
    # For all blocks
    # N = datasets[0][key].shape[1]
    N = datasets[0].shape[1]
    print(N)
    xtraj = np.zeros((2*nBlocks, N))
    for n in range(nBlocks):
        # Weld base to world and set color
        visualizer = weld_base_to_world(visualizer, visualizer.model_index[n], translation=np.array([0., 2.*n, 0.]))
        visualizer = set_block_color(visualizer, visualizer.model_index[n], colors[n])
        # Map the trajectories to the multi-block state
        # xtraj[n,:] = datasets[n][key][0,:N]
        # xtraj[nBlocks+n,:] = datasets[n][key][1,:N]
        xtraj[n,:] = datasets[n][0,:N]
        xtraj[nBlocks+n,:] = datasets[n][1,:N]
    # Add in the terrain and weld
    visualizer = add_terrain(visualizer, terraincolor)
    # Get the common time axis (assume all have the same time axis)
    # time = np.squeeze(datasets[0]['time'])
    time = np.loadtxt('data/slidingblock/erm/t.txt')
    # Create the trajectory
    traj = PiecewisePolynomial.FirstOrderHold(time, xtraj)
    # Visualize
    visualizer.visualize_trajectory(traj)

if __name__ == "__main__":
    print("Hello, world")