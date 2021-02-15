# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
# from trajopt.contactimplicit import ContactImplicitDirectTranscription
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
import utilities as utils

# Create the hopper model with the default flat terrain
plant = TimeSteppingMultibodyPlant(file="systems/urdf/single_legged_hopper.urdf")
plant.Finalize()
# Get the default context
context = plant.multibody.CreateDefaultContext()

# Create a Contact Implicit Trajectory Optimization
trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context)

# Add initial and final state
x0 = np.array([0, 2, 0, 0, 0])
xf = np.array([5, 2, 0, 0, 0])
trajopt.add_state_constraint(knotpoint=0, value=x0)    
trajopt.add_state_constraint(knotpoint=100, value=xf)
# Set all the timesteps to be equal
trajopt.add_equal_time_constraints()
print('Done!')