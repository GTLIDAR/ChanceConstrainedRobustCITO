# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
from trajopt.contactimplicit import ContactImplicitDirectTranscription
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
import utilities as utils
from scipy.special import erfinv
from a1 import create_a1_multibody
# create A1 model
plant, _ = create_a1_multibody()
# plant = TimeSteppingMultibodyPlant(file="systems/A1/A1_description/urdf/a1.urdf")
# Get the default context
context = plant.CreateDefaultContext()
# Create a Contact Implicit Trajectory Optimization
trajopt = ContactImplicitDirectTranscription(plant=plant,
                                            context=context,
                                            num_time_samples=101,
                                            maximum_timestep=0.01,
                                            minimum_timestep=0.01)
print("done!")