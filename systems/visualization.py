"""
Visualization: module for visualizating pyDrake systems with MeshCat

Luke Drnach
January 19, 2021
"""

import numpy as np
from pydrake.all import (PiecewisePolynomial, MultibodyPlant, SceneGraph, ClippingRange, DepthRange, DepthRenderCamera, RenderCameraCore, RenderLabel, MakeRenderEngineVtk, RenderEngineVtkParams, TrajectorySource, MultibodyPositionToGeometryPose)
from pydrake.geometry import DrakeVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import CameraInfo, RgbdSensor

from utilities import FindResource

#TODO: Refactor for cleanliness and readability

class Visualizer():
    def __init__(self, urdf):
        self._create_plant(urdf)

    def _create_plant(self, urdf):
        self.plant = MultibodyPlant(time_step=0.0)
        self.scenegraph = SceneGraph()
        self.plant.RegisterAsSourceForSceneGraph(self.scenegraph)
        self.model_index = Parser(self.plant).AddModelFromFile(FindResource(urdf))
        self.builder = DiagramBuilder()
        self.builder.AddSystem(self.scenegraph)

    def _finalize_plant(self):
        self.plant.Finalize()

    def _add_trajectory_source(self, traj):
        # Create the trajectory source
        source = self.builder.AddSystem(TrajectorySource(traj))
        pos2pose = self.builder.AddSystem(MultibodyPositionToGeometryPose(self.plant, input_multibody_state=True))
        # Wire the source to the scene graph
        self.builder.Connect(source.get_output_port(0), pos2pose.get_input_port())
        self.builder.Connect(pos2pose.get_output_port(), self.scenegraph.get_source_pose_port(self.plant.get_source_id()))

    def _add_renderer(self):
        renderer_name = "renderer"
        self.scenegraph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
        # Add a camera
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer_name, 
                CameraInfo(width=640, height=480, fov_y=np.pi/4),
                ClippingRange(0.01, 10.0),
                RigidTransform()),
            DepthRange(0.01,10.0)
        )
        world_id = self.plant.GetBodyFrameIdOrThrow(self.plant.world_body().index())
        X_WB = xyz_rpy_deg([4,0,0],[-90,0,90])
        sensor = RgbdSensor(world_id, X_PB=X_WB, depth_camera=depth_camera)
        self.builder.AddSystem(sensor)
        self.builder.Connect(self.scenegraph.get_query_output_port(), sensor.query_object_input_port())
    
    def _connect_visualizer(self):
        DrakeVisualizer.AddToBuilder(self.builder, self.scenegraph)
        self.meshcat = ConnectMeshcatVisualizer(self.builder, self.scenegraph, zmq_url="new", open_browser=False)
        self.meshcat.vis.jupyter_cell()

    def _make_visualization(self, stop_time):
        simulator = Simulator(self.builder.Build())
        simulator.Initialize()
        self.meshcat.vis.render_static()
        # Set simulator context
        simulator.get_mutable_context().SetTime(0.0)
        # Start recording and simulate the diagram
        self.meshcat.reset_recording()
        self.meshcat.start_recording()
        simulator.AdvanceTo(stop_time)
        # Publish the recording
        self.meshcat.publish_recording()
        # Render
        self.meshcat.vis.render_static()
        input("View visualization. Press <ENTER> to end")
        print("Finished")

    def visualize_trajectory(self, xtraj=None):
        self._finalize_plant()
        print("Adding trajectory source")
        xtraj = self._check_trajectory(xtraj)
        self._add_trajectory_source(xtraj)
        print("Adding renderer")
        self._add_renderer()
        print("Connecting to MeshCat")
        self._connect_visualizer()
        print("Making visualization")
        self._make_visualization(xtraj.end_time())
        
    def _check_trajectory(self, traj):
        if traj is None:
            plant_context = self.plant.CreateDefaultContext()
            pose = self.plant.GetPositions(plant_context)
            pose = np.column_stack((pose, pose))
            pose = zero_pad_rows(pose, self.plant.num_positions() + self.plant.num_velocities())
            return PiecewisePolynomial.FirstOrderHold([0., 1.], pose)
        elif type(traj) is np.ndarray:
            if traj.ndim == 1:
                traj = np.column_stack(traj, traj)
            traj = zero_pad_rows(traj, self.plant.num_positions() + self.plant.num_velocities())
            return PiecewisePolynomial.FirstOrderHold([0.,1.], traj)
        elif type(traj) is PiecewisePolynomial:
            breaks = traj.get_segment_times()
            values = traj.vector_values(breaks)
            values = zero_pad_rows(values, self.plant.num_positions() + self.plant.num_velocities())
            return PiecewisePolynomial.FirstOrderHold(breaks, values)
        else:
            raise ValueError("Trajectory must be a piecewise polynomial, an ndarray, or None")

# Helper methods
def xyz_rpy_deg(xyz, rpy_deg):
    """Defines a pose"""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi/180), xyz)

def zero_pad_rows(val, totalrows):
    """Helper function to zeropad an array to have a desired number of rows"""
    numrows = val.shape[0]
    if totalrows > numrows:
        return np.pad(val, ((0, totalrows-numrows), (0,0)), mode="constant")
    elif totalrows < numrows:
        return val[0:totalrows, :]
    else:
        return val

if __name__ == "__main__":
   file = "systems/urdf/single_legged_hopper.urdf"

   angle = 0.5 
   height = np.cos(0.5) * 2
    # Add initial and final state
   x0 = np.array([0, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
   vis = Visualizer(file)
   vis.visualize_trajectory()