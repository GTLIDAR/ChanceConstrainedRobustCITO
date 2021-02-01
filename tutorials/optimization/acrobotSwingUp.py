"""
acrobotSwingUp: example direct collocation trajectory optimization in pyDrake

acrobotSwingUp creates and solves the trajectory optimization problem of swinging the acrobot from the downward position to the upward position. The nonlinear program is developed using the DirectTranscription class and solved using SNOPT.

adapted from Direct Collocation for the Acrobot tutorial on Russ Tedrake's Underactuated Robotics course webpage: http://underactuated.mit.edu/trajopt.html

Luke Drnach
September 30, 2020
"""
from math import pi 
import matplotlib.pyplot as plt 
import numpy as np 
import timeit
# Import utilities from pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.all import (MultibodyPlant, PiecewisePolynomial, DirectCollocation, DiagramBuilder, SceneGraph, PlanarSceneGraphVisualizer, Simulator, TrajectorySource, MultibodyPositionToGeometryPose, AddMultibodyPlantSceneGraph, Solve, RigidTransform)
from pydrake.multibody.parsing import Parser
from pydrake.solvers.snopt import SnoptSolver

# Create a Multibody plant model from the acrobot
plant = MultibodyPlant(time_step=0.0)
scene_graph = SceneGraph()
plant.RegisterAsSourceForSceneGraph(scene_graph)
# Find and load the Acrobot URDF 
# Note that we cannot include collision geometry in the URDF, otherwise setting up the visualization will fail.
acro_file = FindResourceOrThrow("drake/examples/acrobot/Acrobot_no_collision.urdf")
Parser(plant).AddModelFromFile(acro_file)
# Weld the base frame to the world frame
base_frame = plant.GetBodyByName("base_link").body_frame()
world_frame = plant.world_frame()
plant.WeldFrames(world_frame, base_frame, RigidTransform())
# Finalize the plant
plant.Finalize()

# Create the default context
context0 = plant.CreateDefaultContext()
# Create a direct collocation problem
prog = DirectCollocation(
    plant,
    context0,
    num_time_samples=21,
    maximum_timestep=0.20,
    minimum_timestep=0.05,
    input_port_index=plant.get_actuation_input_port().get_index())
prog.AddEqualTimeIntervalsConstraints()
# Add initial and final state constraints
x0 = [0, 0, 0, 0]
xf = [pi, 0, 0,0]
prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())
# Add running cost
u = prog.input()
R = 10
prog.AddRunningCost(R * u[0] ** 2)
# Add final cost
prog.AddFinalCost(prog.time())
# Create an initial guess at the trajectory
init_x = PiecewisePolynomial.FirstOrderHold([0, 10.0], np.column_stack((x0, xf)))
init_u = PiecewisePolynomial.FirstOrderHold([0, 10.0], np.zeros(shape=(1,2)))
prog.SetInitialTrajectory(traj_init_u=init_u, traj_init_x=init_x)
# Set SNOPT options
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Iterations Limit", 1000)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-4)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-4)
prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
solver = SnoptSolver()
# Solve the problem
print("Solving trajectory optimization")
start = timeit.default_timer()
result = solver.Solve(prog)
stop = timeit.default_timer()
print("Elapsed Time: ", stop-start)
# Get the details  of the solution
print("Optimization successful? ", result.is_success())
print('solver is: ', result.get_solver_id().name())
print('optimal cost = ', result.get_optimal_cost())
# Get the exit code from SNOPT
details = result.get_solver_details()
print('SNOPT Exit Status: ',details.info)

# Unpack the trajectories
u_traj = prog.ReconstructInputTrajectory(result)
x_traj = prog.ReconstructStateTrajectory(result)    

time = np.linspace(u_traj.start_time(), u_traj.end_time(), 101)
u_lookup = np.vectorize(u_traj.value)
u = u_lookup(time)
x = np.hstack([x_traj.value(t) for t in time])
# Plot the trajectory
plt.figure()
plt.subplot(3,1,1)
plt.plot(time, x[0,:],label="shoulder")
plt.plot(time, x[1,:],label="elbow")
plt.legend()
plt.ylabel('Positions (rad)')
plt.subplot(3,1,2)
plt.plot(time, x[2,:])
plt.plot(time, x[3,:])
plt.ylabel('Velocities (rad/s)')
plt.subplot(3,1,3)
plt.plot(time, u)
plt.ylabel('Torque (Nm)')
plt.xlabel('Time (s)')
plt.show()

# Visualize the results
# A Diagram is a collection of systems in a directed graph. To visualize the results, we need to attach the solution trajectory, a SceneGraph, and a visualizer together.
# After constructing the Diagram, we need to
#   1. Add systems to the Diagram
#   2. Connect the systems in the Diagram

# AddSystem is the generic method to add components to the Diagram.
# TrajectorySource is a type of System whose output is the value of a trajectory at a time in the system's context
builder = DiagramBuilder()
source = builder.AddSystem(TrajectorySource(x_traj))
builder.AddSystem(scene_graph)
to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant, input_multibody_state=True))
# Wire the ports of hte systems together
builder.Connect(source.get_output_port(0), to_pose.get_input_port())
builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))
# Add a visualizer
T_VW = np.array([[1., 0., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
visualizer = builder.AddSystem(PlanarSceneGraphVisualizer(scene_graph, T_VW=T_VW, xlim=[-4.,4.], ylim=[-4., 4.], show=True))
builder.Connect(scene_graph.get_pose_bundle_output_port(), visualizer.get_input_port(0))
# build and run the simulator
simulator = Simulator(builder.Build())
simulator.Initialize()
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(x_traj.end_time())
