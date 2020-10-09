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
from pydrake.all import MultibodyPlant, DiagramBuilder, AddMultibodyPlantSceneGraph
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
plant.Finalize()
context = plant.CreateDefaultContext()
# We will also need a scene graph inspector
inspector = scene_graph.model_inspector()
# Based on the URDF, Box should have 8 collision geometries
print(f"Box has {plant.num_collision_geometries()} collision geometries")
# Locate the collision geometries and contact points
# First get the rigid body elements
body_inds = plant.GetBodyIndices(box)
body = plant.get_body(body_inds[0])
collisionIds = plant.GetCollisionGeometriesForBody(body)

for id in collisionIds:
    # Get the frame in which the collision geometry resides
    frameId = inspector.GetFrameId(id)
    frame_name = inspector.GetName(frameId)
    # Get the pose of the collision geometry in the frame
    R = inspector.GetPoseInFrame(id)
    # Print the pose in homogeneous coordinates
    print(f"Collision geometry in frame {frame_name} with pose \n{R.GetAsMatrix4()}")