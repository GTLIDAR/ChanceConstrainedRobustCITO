import numpy as np
from utilities import load, FindResource  
from systems.visualization import Visualizer
from pydrake.all import PiecewisePolynomial
from matlab_erm.visualizeBlockCore import weld_base_to_world, colors, add_terrain

def set_cart_colors(vis, model_index, color=np.array([1., 1., 1., 1.])):
    """Change the colors of all the links in the cart"""
    bodies = vis.plant.GetBodyIndices(model_index)
    bodies = bodies[1:] #Exclude the rail
    for body in bodies:
        vis.setBodyColor(body, color)
    return vis

def visualize_multicart(datasets, colors, terraincolor=None):
    """Visualize multiple carts at once"""
    nCarts = len(datasets)
    # Create a cart visualizer
    cartfile = "matlab_erm/urdf/contactcart.urdf"
    visualizer = Visualizer(FindResource(cartfile))
    for n in range(1, nCarts):
        visualizer.addModelFromFile(FindResource(cartfile), name="Cart"+str(n))
    # Weld all carts to the world frame
    nX, nT = datasets[0]['xtraj'].shape
    xtraj = np.zeros((nX*nCarts, nT))
    for n in range(nCarts):
        # Weld the base and set color
        visualizer = weld_base_to_world(visualizer, visualizer.model_index[n], translation=np.array([0., 0*n, 0.]))
        visualizer = set_cart_colors(visualizer, visualizer.model_index[n], colors[n])
        for k in range(nX):
            xtraj[n + k*nCarts,:] = datasets[n]['xtraj'][k,:nT]
    # Add in the terrain and weld
    visualizer = add_terrain(visualizer, terraincolor)
    # Get the common time axis
    time = np.squeeze(datasets[0]['time'])
    # Create the trajectory
    traj = PiecewisePolynomial.FirstOrderHold(time, xtraj)
    # Visualize
    visualizer.visualize_trajectory(traj)

if __name__ == "__main__":
    ermdata = load("data/python_erm/CartERM.pkl")
    nominal = load("data/python_erm/CartNominal.pkl")
    for data  in ermdata:
        print(f"ERM Sigma = {data['sigma']}")
    dataset = [nominal[0], ermdata[4], ermdata[5], ermdata[6]]
    colornames=["blue","purple","yellow","red"]
    datacolors = [colors[name] for name in colornames]
    # Visualize
    visualize_multicart(dataset, datacolors)
    # Print out the sigma values and colors
    print(f"Visualized nominal in {colornames[0]}")
    for n in range(1,4):
        print(f"Visualized ERM with sigma = {ermdata[n+3]['sigma']} in {colornames[n]}")