import numpy as np
from math import pi

from Systems.Rod import Rod
from TimeStepping.RigidContactModel import RigidContactModel
from TimeStepping.Terrain import FlatTerrain2D

# Create rigid body model with contact
model = RigidContactModel(plant=Rod(m=1, l=0.5, r=0.05, J=0.002), terrain=FlatTerrain2D())
# Run a test simulation
X0 = np.array([0.0, 1.0, pi/6, 0.0, 0.0, 4.0])
t,x,f = model.simulate(X0, T=1.5, h=0.01)
# Plot the trajectory
model.plant.plotTrajectory(t,x,f)