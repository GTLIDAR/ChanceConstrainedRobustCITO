"""
collision.py: Tutorial on accessing collision properties from MultibodyPlant in Drake

Luke Drnach
October 7, 2020
"""
# Python imports
from math import pi
from os import path 
from sys import exit
# PyDrake Imports 
from pydrake.common import FindResourceOrThrow
from pydrake.all import MultibodyPlant, DiagramBuilder, AddMultibodyPlantSceneGraph, RigidTransform, JacobianWrtVariable
from pydrake.multibody.parsing import Parser

# Import the model from a URDF file
model_file = "Systems/urdf/fallingBox.urdf"
if not path.isfile(model_file):
    exit(f"Path {model_file} not found")
else:
    model_file = path.abspath(model_file)
print(model_file)

# Set up the RigidBodyPlant model in Drake - Note that without the DiagramBuilder and SceneGraph, pydrake return that there are no collision geometries
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.001)
box = Parser(plant).AddModelFromFile(model_file)
body_inds = plant.GetBodyIndices(box)
base_frame = plant.get_body(body_inds[0]).body_frame()
world_frame = plant.world_frame()
plant.WeldFrames(world_frame, base_frame, RigidTransform())
plant.Finalize()
context = plant.CreateDefaultContext()
print(f"fallingbox has {plant.num_positions()} position coordinates")
# We will also need a scene graph inspector
inspector = scene_graph.model_inspector()
# Based on the URDF, Box should have 8 collision geometries
print(f"Box has {plant.num_collision_geometries()} collision geometries")
# Locate the collision geometries and contact points
# First get the rigid body elements
collisionIds = []
body_inds = plant.GetBodyIndices(box)
for ind in body_inds:
    body = plant.get_body(ind)
    collisionIds.append(plant.GetCollisionGeometriesForBody(body))

# Flatten the list
collisionIds = [id for id_list in collisionIds for id in id_list if id_list]
# Move the box vertically upward to make the example slightly more interesting
plant.SetPositions(context, [0, 0, 1, 0, 0, 0])
# Loop over the collision identifiers
for id in collisionIds:
    # Get the frame in which the collision geometry resides
    frameId = inspector.GetFrameId(id)
    frame_name = inspector.GetName(frameId)
    # Get the pose of the collision geometry in the frame
    R = inspector.GetPoseInFrame(id)
    # Print the pose in homogeneous coordinates
    print(f"Collision geometry in frame {frame_name} with pose \n{R.GetAsMatrix4()}")
    # Note that the frameId is NOT the same as the FrameIndex, and cannot be used to the get the frame from MultibodyPlant. We can strip the frame name from the ID and use that instead
    name = frame_name.split("::")
    body_frame = plant.GetFrameByName(name[-1])
    # Then we can calculate the position in world coordinates
    world_pt = plant.CalcPointsPositions(context, body_frame, R.translation(), world_frame)
    print(f"In world coordinates, the collision point is\n {world_pt}")    
    # Finally, we can calculate the Jacobian for the contact point
    Jc = plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable.kQDot, body_frame, R.translation(), world_frame, world_frame)
    print(f"The Jacobian for the contact point in world coordinates is \n{Jc}")