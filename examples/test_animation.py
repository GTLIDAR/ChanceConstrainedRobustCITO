from IPython.display import HTML
from matplotlib import animation
import numpy as np
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import (
    PlanarSceneGraphVisualizer)

def run_pendulum_example(duration=1., playback=True, show=True):
    """
    Runs a simulation of a pendulum.

    Arguments:
        duration: Simulation duration (sec).
        playback: Enable pyplot animations to be produced.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.)
    parser = Parser(plant)
    parser.AddModelFromFile(FindResourceOrThrow(
        "drake/examples/pendulum/Pendulum.urdf"))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.Finalize()

    pose_bundle_output_port = scene_graph.get_pose_bundle_output_port()
    # T_VW = np.array([[1., 0., 0., 0.],
    #                  [0., 0., 1., 0.],
    #                  [0., 0., 0., 1.]])
    visualizer = builder.AddSystem(PlanarSceneGraphVisualizer(
        scene_graph,
        xlim=[-1.2, 1.2], ylim=[-1.2, 1.2], show=show))
    builder.Connect(pose_bundle_output_port, visualizer.get_input_port(0))
    if playback:
        visualizer.start_recording()

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)

    # Fix the input port to zero.
    plant_context = diagram.GetMutableSubsystemContext(
        plant, simulator.get_mutable_context())
    plant.get_actuation_input_port().FixValue(
        plant_context, np.zeros(plant.num_actuators()))
    plant_context.SetContinuousState([0.5, 0.1])
    simulator.AdvanceTo(duration)

    if playback:
        visualizer.stop_recording()
        ani = visualizer.get_recording_as_animation()
        return ani
    else:
        return None

ani = run_pendulum_example(playback=True)

HTML(ani.to_jshtml())
if animation.writers.is_available("ffmpeg"):
    display(HTML(ani.to_html5_video()))
