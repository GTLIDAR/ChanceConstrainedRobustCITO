# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
# from pydrake.all import ClippingRange
# ClippingRange(0.01, 10.0)
# from pydrake.all import ClippingRange
# from trajopt.contactimplicit import ContactImplicitDirectTranscription
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
from systems.visualization import Visualizer
import utilities as utils
from plot_hopper import plot
from pydrake.all import PiecewisePolynomial
# Create the hopper model with the default flat terrain
_file = "systems/urdf/single_legged_hopper.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
plant.Finalize()
# Get the default context
context = plant.multibody.CreateDefaultContext()
num_step = 301
angle = 0.5 
height = np.cos(0.5) * 2
# Add initial and final state
x0 = np.array([0, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
xf = np.array([4, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
tolerance = 1e-6
slack = 10
for i in range(6):
    
    
    # Create a Contact Implicit Trajectory Optimization
    trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                                context=context,
                                                num_time_samples=num_step)
    # x0 = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    # xf = np.array([4, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    trajopt.add_state_constraint(knotpoint=0, value=x0)    
    trajopt.add_state_constraint(knotpoint=300, value=xf)
    # Set all the timesteps to be equal
    trajopt.add_equal_time_constraints()
    print("Current slack is ", slack)
    print("Current tolerance is ", tolerance)
    trajopt.set_slack(slack)
    slack = slack/10
    if i is 5:
        slack = 0
    # Add a final cost on the total time
    cost = lambda h: np.sum(h)
    trajopt.add_final_cost(cost, vars=[trajopt.h], name="TotalTime")
    # Set the initial trajectory guess
    if i is 0:
        u_init = np.zeros(trajopt.u.shape)
        x_init = np.zeros(trajopt.x.shape)
        for n in range(0, x_init.shape[0]):
            x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=num_step)
        l_init = np.zeros(trajopt.l.shape)
        x_init = np.loadtxt('data/single_legged_hopper/nominal_3/x_s.txt')
        u_init = np.loadtxt('data/single_legged_hopper/nominal_3/u_s.txt')
        u_init = u_init.reshape(trajopt.u.shape)
        l_init = np.loadtxt('data/single_legged_hopper/nominal_3/l_s.txt')
        t_init = np.loadtxt('data/single_legged_hopper/nominal_3/t.txt')

    else: 
        # Add a running cost on the controls and state
        # add costs after the first iteration
        # R=  0.01*np.diag([1,1,1])
        # b = np.zeros((3,))
        # trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
        # Q = np.diag([1,10,10,100,100,1,1,1,1,1])
        # trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
        # load saved traj
        x_init = np.loadtxt('data/single_legged_hopper/nominal_3/x{n}.txt'.format(n = i-1))
        u_init = np.loadtxt('data/single_legged_hopper/nominal_3/u{n}.txt'.format(n = i-1))
        u_init = u_init.reshape(trajopt.u.shape)
        l_init = np.loadtxt('data/single_legged_hopper/nominal_3/l{n}.txt'.format(n = i-1))
        t_init = np.loadtxt('data/single_legged_hopper/nominal_3/t.txt')
    trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
    # Get the final program, with all costs and constraints
    prog = trajopt.get_program()
    # Set the SNOPT solver options
    
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", tolerance)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", tolerance)
    # tolerance = tolerance/10
    if i is 0:
        prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e5)
    else:
        prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 1)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1e5)
    solver = SnoptSolver()
    # trajopt.enable_cost_display(display='figure')
    if not utils.CheckProgram(prog):
        quit()

    print("Solving trajectory optimization")
    start = timeit.default_timer()
    result = solver.Solve(prog)
    stop = timeit.default_timer()
    print(f"Elapsed time: {stop-start}")
    # Print the details of the solution
    print(f"Optimization successful? {result.is_success()}")
    print(f"Solved with {result.get_solver_id().name()}")
    print(f"Optimal cost = {result.get_optimal_cost()}")
    # Get the exit code from SNOPT
    print(f"SNOPT Exit Status {result.get_solver_details().info}: {utils.SNOPT_DECODER[result.get_solver_details().info]}")
    # Unpack and plot the trajectories
    x = trajopt.reconstruct_state_trajectory(result)
    u = trajopt.reconstruct_input_trajectory(result)
    l = trajopt.reconstruct_reaction_force_trajectory(result)
    t = trajopt.get_solution_times(result)
    np.savetxt('data/single_legged_hopper/nominal_3/x{n}.txt'.format(n = i), x, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/nominal_3/u{n}.txt'.format(n = i), u, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/nominal_3/l{n}.txt'.format(n = i), l, fmt = '%1.3f')
    np.savetxt('data/single_legged_hopper/nominal_3/t.txt'.format(n = i), t, fmt = '%1.3f')
    

    plot(x, u, l, t)
x = PiecewisePolynomial.FirstOrderHold(t, x)
vis = Visualizer(_file)
body_inds = vis.plant.GetBodyIndices(vis.model_index)
base_frame = vis.plant.get_body(body_inds[0]).body_frame()
vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())

vis.visualize_trajectory(x)

print('Done!')