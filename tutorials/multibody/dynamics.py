"""
dynamics.py: tutorial on accessing MultibodyPlant dynamics using Acrobot

Luke Drnach
October 7, 2020
"""
#%% [markdown]
# ## AcrobotDynamics: Tutorial on accessing MultibodyPlant dynamics using Acrobot
#   
#   The following code demonstrates how to access the components of the manipulator dynamics equations from Drake's MultibodyPlant class, using the Acrobot as an example system. Note that not all functionality from the C++ Drake API is available in the Python API; however, the Python API is sufficiently powerful to give access to the all componetents of the manipulator dynamics equations

from math import pi 
# Import utilities from pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.all import MultibodyPlant
from pydrake.multibody.parsing import Parser
#from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.tree import MultibodyForces
#%% [markdown]
#   # Loading a the acrobot model from a URDF
#   
#   The Acrobot model is supplied with the Drake docker image. We can load it and directly create a Drake MultibodyPlant. To fully utilitze the MultibodyPlant, we will also need to create a Context, which stores values associated with the MultibodyPlant such as the configuration and velocity variables.

# Find and load the Acrobot URDF 
acro_file = FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf")
# Create a Multibody plant model from the acrobot
plant = MultibodyPlant(0.0)
acrobot = Parser(plant).AddModelFromFile(acro_file)
plant.Finalize()
# The CONTEXT of the system stores information, such as the state, about the multibody plant. The context is necessary to calculate values such as the inertia matrix
context = plant.CreateDefaultContext()
##% [markdown]
# 
#  The default context is not very interesting, so let's set some nonzero values for the configuration and velocity variables
q = [pi/4, pi/4]
v = [ 0.1, -0.1]
plant.SetPositions(context, q)
plant.SetVelocities(context, v)
#%% [markdown]
#   # DYNAMICS

# We can get the mass matrix via inverse dynamics (note that CalcMassMatrix is not available in pyDrake):
M = plant.CalcMassMatrixViaInverseDynamics(context)
print('Acrobot mass matrix at q = ',q,' is M = ')
print(M)
# We can also get the combined coriolis, centripetal, and gyroscopic effects:
Cv = plant.CalcBiasTerm(context)
print("Acrobot bias term at q = ", q, "and v = ",v ,"is Cv = ")
print(Cv)
# We can separately get the gravitational effects
N = plant.CalcGravityGeneralizedForces(context)
print("Acrobot gravitational generalized forces at q = ", q," is N = ")
print(N)
# Evaluating the controls
nU = plant.num_actuated_dofs()
nA = plant.num_actuators()
print('Acrobot has ', nU, ' actuated joint(s) and ', nA, ' actuator(s)')
# We can get the actuation matrix, a permutation  matrix as:
B = plant.MakeActuationMatrix()
print('The acutator selection matrix for acrobot is B = ')
print(B)
# Note that calculating the dynamics in this fashion is not computationally efficient. It would be more efficient to use plant.CalcInverseDynamics instead, given the generalized acceleration and applied forces

# Create empty generalized applied forces
forces = MultibodyForces(plant)
forces.SetZero()
# Create some generalized accelerations
dv = [0.2, 0.6]
# Do inverse dynamics to find the generalized forces needed to achieve these accelerations
# NOTE: INVERSE DYNAMICS DOES NOT AUTOMATICALLY ENCODE THE GRAVITATIONAL FORCES
tau = plant.CalcInverseDynamics(context, dv, forces)
print(f"Inverse dynamics without gravity f = {tau}")
# To encode generalized forces - including gravity - we can add them in after the fact, or add them in to the MultibodyForces
force = forces.mutable_generalized_forces()
force[:] = N
tau_2 = plant.CalcInverseDynamics(context,dv, forces)
print(f"Inverse dynamics with gravity f = {tau_2}")

