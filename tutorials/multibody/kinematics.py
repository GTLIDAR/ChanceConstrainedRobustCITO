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

# We can get the world body and world frame
wbody = plant.world_body()
wframe = plant.world_frame()
# We can get the bodies and the names of the bodies from the plant
body_indices = plant.GetBodyIndices(acrobot)
for n in range(0, len(body_indices)):
    name = plant.get_body(body_indices[n]).name()
    print(f"Acrobot body {n} is called {name}")
    body_frame = plant.get_body(body_indices[n]).body_frame()
    frame_name = body_frame.name()
    print(f"Body {name} has a frame called {frame_name}")
    # We can get the frame index for the body frame as
    frame_idx = body_frame.index()
    # Then we can calculate the location of points in the body frames in world coordinates
    pW = plant.CalcPointsPositions(context, body_frame, [0., 0., 0.], wframe)
    print(f"The origin of frame {frame_name} in world coordinates is")
    print(pW)
    # Alternatively, we can calculate the pose of the body in world coordinates
    pose = plant.EvalBodyPoseInWorld(context, plant.get_body(body_indices[n]))
    print(f"The pose of body {name} in world coordinates is")
    print(pose.GetAsMatrix4())
    # We can also calculate the Jacobian of the frame in world coordinates
    J = plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable.kQDot, body_frame, [0.,0.,0.], wframe, wframe)
    print(f"The Jacobian for the origin of body {name} in world coordinates is")
    print(J)