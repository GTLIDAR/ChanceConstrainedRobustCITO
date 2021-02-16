import numpy as np
from pydrake.all import (PiecewisePolynomial, MultibodyPlant, SceneGraph, RenderLabel, MakeRenderEngineVtk, RenderEngineVtkParams, TrajectorySource, MultibodyPositionToGeometryPose)
# from pydrake.geometry import DrakeVisualizer
# from pydrake.geometry import drake_visualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import CameraInfo, RgbdSensor

from utilities import FindResource
def visualize(urdf, xtraj):
    plant = MultibodyPlant(time_step = 0.0)
    scenegraph = SceneGraph()
    plant.RegisterAsSourceForSceneGraph(self.scenegraph)
    model_index = Parser(plant).AddModelFromFile(FindResource(urdf))
    builder = DiagramBuilder()
    builder.AddSystem(scenegraph)
    plant.Finalize()
