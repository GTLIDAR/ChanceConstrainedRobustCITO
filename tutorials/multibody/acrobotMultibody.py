"""
acrobotMultibody.py: Tutorial script on working with MultibodyPlant in Drake, using the Acrobot as an example

Luke Drnach
October 2, 2020
"""
from math import pi 
# Import utilities from pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.all import MultibodyPlant, JacobianWrtVariable
from pydrake.multibody.parsing import Parser

# Find and load the Acrobot URDF 
acro_file = FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf")
# Create a Multibody plant model from the acrobot
plant = MultibodyPlant(0.0)
acrobot = Parser(plant).AddModelFromFile(acro_file)
plant.Finalize()

# The CONTEXT of the system stores information, such as the state, about the multibody plant. The context is necessary to calculate values such as the inertia matrix
context = plant.CreateDefaultContext()

## STATES AND DIMENSIONS
# Evaluating and updating the position and velocity variables
# Get and print the number of position and velocity variables
nQ = plant.num_positions()
nV = plant.num_velocities()
print("The acrobot has ", nQ, " generalized positions and ", nV, " generalized velocities")
# We can get the values of the positions and velocities from the CONTEXt
q = plant.GetPositions(context)
v = plant.GetVelocities(context)
print("The acrobot default configuration is q = ", q)
print("The acrobot default velocity is v = ", v)
# We can also get the full state, x = [q, v]
x = plant.GetPositionsAndVelocities(context)
print('Acrobot full state x = ', x)
# Drake implements a mapping between the generalized positions and velocities. The default mapping is:
dq = plant.MapVelocityToQDot(context, v)
print("The default acrobot configuration rates are dq = ", dq)

# We can set the values for the positions and velocities as well:
plant.SetPositionsAndVelocities(context, [pi/4, pi/4, 0.1, -0.1 ])
# Then we can get the new values of the position and velocity
q = plant.GetPositions(context)
v = plant.GetVelocities(context)
# And compare against the configuration rates
dq = plant.MapVelocityToQDot(context, v)
print("The modified acrobot configuration is q = ", q, " and velocity v = ", v)
print("The acrobot configuration rates are dq =", dq)
# Finally, we can check for lower and upper joint limits
qlow = plant.GetPositionLowerLimits()
qhigh = plant.GetPositionUpperLimits()
print(f'The acrobot has upper limits at {qhigh} and lower limits at {qlow}')

## KINEMATICS and GEOMETRY
# Let's start with the frames in Acrobot
nF = plant.num_frames()
nJ = plant.num_joints()
print(f"Acrobot has {nF} frames and {nJ} joints")

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


# We CANNOT Get the joint indices using plant.GetJointIndices(acrobot).  The method GetJointIndices does not exist in pydrake
 # Note - we've only explored 4 of the 8 frames in Acrobot. 

# Joints and Actuators
# Note that we can get the indices of the joints connected to actuators, provided we can get the indices/names of the actuators
actuator = plant.GetJointActuatorByName('elbow')
# We can check for effort limits
print("Actuator ", actuator.name(), " has effort limit ", actuator.effort_limit())
# We can get the joint attached to the actuator
joint = actuator.joint()
# Then we can query the parent body, the child body, and their frames
joint_child = joint.frame_on_child().name()
joint_parent = joint.frame_on_parent().name()
joint_name = joint.name()
print(f"Joint {joint_name} connects parent body {joint_parent} to child body {joint_child}")

# Contact Geometry
