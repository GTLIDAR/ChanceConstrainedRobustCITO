"""
kinematics.py: Tutorial script on accessing kinematics operations in pyDrake's RigidbodyPlant
Luke Drnach
Octoer 7, 2020
"""
#%% [markdown]
## Kinematics: Accessing frames and forward kinematics in pyDrake
#   Here we demonstrate how to access body-fixed frames from MultibodyPlant and how to use the frames to perform forward kinematics in pyDrake. We use an acrobot, an underactuated two-link pendulum, as our example system. The description of the acrobot is in a URDF file included in the Drake installation. 
#   To begin, we'll need to:  
#       1. import a few resources from Drake
#       2. Create the MultibodyPlant using the Acrobot URDF
#       3. Create a default *context* to store the model parameters

from math import pi
from pydrake.common import FindResourceOrThrow
from pydrake.all import MultibodyPlant, JacobianWrtVariable
from pydrake.multibody.parsing import Parser

# Create the MultibodyPlant from the URDF
# Find and load the Acrobot URDF 
acro_file = FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf")
# Create a Multibody plant model from the acrobot
plant = MultibodyPlant(0.0)
acrobot = Parser(plant).AddModelFromFile(acro_file)
plant.Finalize()
# Get the default context
context = plant.CreateDefaultContext()

#%% [markdown]
#
#   The acrobot as 4 states: the two relative angles of the links and their respective joint rates. The default context sets all state to zero. To make the kinematics more interesting, we can set a nonzer state vector.
plant.SetPositionsAndVelocities(context, [pi/4, pi/4, 0.1, -0.1 ])

#%% [markdown]
#
#   ### Kinematics