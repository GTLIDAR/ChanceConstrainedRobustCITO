"""
Tutorial script on visualizing a simulation in pyDrake

Adapted from pyplot_animation_multibodyplant.ipynb Drake tutorial (https://drake.mit.edu/)

Luke Drnach
October 8, 2020
"""
from matplotlib import animation
import numpy as np 

from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import PlanarSceneGraphVisualizer



def pendulum_example(duration=1., playback=True, show=True):
    """
        Simulate the pendulum
        
        Arguments:
            duration: Simulation duration (sec) 
            playback: enable pyplot animations
    """
    # To make a visualization, we have to attach a multibody plant, a scene graph, and a visualizer together. In Drake, we can connect all these systems together in a Diagram.
    builder = DiagramBuilder()
    # AddMultibodyPlantSceneGraph: Adds a multibody plant and scene graph to the Diagram, and connects their geometry ports. The second input is the timestep for MultibodyPlant
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.)
    # Now we create the plant model from a file
    parser = Parser(plant)
    parser.AddModelFromFile(FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.Finalize()
    # The SceneGraph port that communicates with the visualizer is the pose bundle output port. A PoseBundle is just a set of poses in SE(3) and a set of frame velocities, expressed in the world frame, used for rendering.
    pose_bundle_output_port = scene_graph.get_pose_bundle_output_port()
    # T_VW is the projection matrix from view coordinates to world coordinates
    T_VW = np.array([[1., 0., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    # Now we add a planar visualizer to the the diagram, so we can see the results
    visualizer = builder.AddSystem(PlanarSceneGraphVisualizer(scene_graph, T_VW=T_VW, xlim=[-1.2, 1.2], ylim=[-1.2, 1.2], show=show))
    # finally, we must connect the scene_graph to the visualizer so they can communicate
    builder.Connect(pose_bundle_output_port, visualizer.get_input_port(0))

    if playback:
        visualizer.start_recording()
    # To finalize the diagram, we build it
    diagram = builder.Build()
    # We create a simulator of our diagram to step through the diagram in time
    simulator = Simulator(diagram)
    # Initialize prepares the simulator for simulation
    simulator.Initialize()
    # Slow down the simulator to realtime. Otherwise it could run too fast
    simulator.set_target_realtime_rate(1.)
    # To set initial conditions, we modify the mutable simulator context (we could do this before Initialize)
    plant_context = diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context())
    plant_context.SetContinuousState([0.5,0.1])
    # Now we fix the value of the actuation to get an unactuated simulation
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros([plant.num_actuators()]))
    # Run the simulation to the specified duration
    simulator.AdvanceTo(duration)
    # Return an animation, if one was made
    if playback:
        visualizer.stop_recording()
        ani = visualizer.get_recording_as_animation()
        return ani
    else:
        return None

if __name__ == "__main__":
    pendulum_example(playback=False)
    