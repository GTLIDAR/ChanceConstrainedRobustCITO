# Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
# from pydrake.all import ClippingRange
# ClippingRange(0.01, 10.0)
# from pydrake.all import ClippingRange
from trajopt.contactimplicit import OptimizationOptions, ContactImplicitDirectTranscription
from trajopt.constraints import NCCImplementation, NCCSlackType
from trajopt.robustContactImplicit import ChanceConstrainedContactImplicit
from systems.timestepping import TimeSteppingMultibodyPlant
from pydrake.solvers.snopt import SnoptSolver
from systems.visualization import Visualizer
import utilities as utils
from plot_hopper import plot
from pydrake.all import PiecewisePolynomial, RigidTransform
# Create the hopper model with the default flat terrain
_file = "systems/urdf/single_legged_hopper.urdf"
plant = TimeSteppingMultibodyPlant(file= _file)
body_inds = plant.multibody.GetBodyIndices(plant.model_index)
base_frame = plant.multibody.get_body(body_inds[0]).body_frame()
plant.multibody.WeldFrames(plant.multibody.world_frame(), base_frame, RigidTransform())
plant.Finalize()
# Get the default context
context = plant.multibody.CreateDefaultContext()
num_step = 101
height = 1.5
angle = np.arccos(0.75/1)
# Add initial and final state
x0 = np.array([0, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
xf = np.array([4, height, -angle, 2*angle, -angle, 0, 0, 0, 0, 0])
tolerance = 1e-6
slack = 100
i = 0
options = OptimizationOptions()
options.ncc_implementation = NCCImplementation.NONLINEAR
options.slacktype = NCCSlackType.CONSTANT_SLACK
# trajopt = ContactImplicitDirectTranscription(plant=plant,
#                                             context=context,
#                                             num_time_samples=num_step,
#                                             minimum_timestep = 0.03,
#                                             maximum_timestep = 0.03,
#                                             options = options)
trajopt = ChanceConstrainedContactImplicit(plant=plant,
                                            context=context,
                                            num_time_samples=num_step,
                                            maximum_timestep=0.03,
                                            minimum_timestep=0.03,
                                            # chance_param= chance_params,
                                            # distance_param= distance_erm_params,
                                            # friction_param= np.array([0.01, 0.01, 1e5]),
                                            # optionERM = uncertainty_option,
                                            # optionCC = cc_option,
                                            options = options)
while slack is not 0:
    if slack > 1e-4:
        slack = slack/10
    else:
        slack = 0
    # tolerance = tolerance/10
    print("iteration", i)
    print("Current slack is ", slack)
    print("Current tolerance is ", tolerance)
    # Create a Contact Implicit Trajectory Optimization
    trajopt.set_slack(slack)
    trajopt.add_state_constraint(knotpoint=0, value=x0)    
    trajopt.add_state_constraint(knotpoint=num_step - 1, value=xf)
    # Set all the timesteps to be equal
    # trajopt.add_equal_time_constraints()
    trajopt.add_equal_time_constraints()
    
    # Add a final cost on the total time
    cost = lambda h: np.sum(h)
    # trajopt.add_final_cost(cost, vars=[trajopt.h], name="TotalTime")
    # Set the initial trajectory guess
    if i is 0:
        first = False
        u_init = np.zeros(trajopt.u.shape)
        for n in range(0, u_init.shape[0]):
            u_init[n,:] = np.linspace(start=0, stop=100, num=101)
        x_init = np.zeros(trajopt.x.shape)
        for n in range(0, x_init.shape[0]):
            x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=101)
        l_init = np.ones(trajopt.l.shape)
        jl_init = np.zeros(trajopt.jl.shape)
        s_init = np.zeros(trajopt.slacks.shape)
        trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj = jl_init, straj = s_init)
        # x_init = np.loadtxt('data/single_legged_hopper/nominal_3/x0.txt')
        # u_init = np.loadtxt('data/single_legged_hopper/nominal_3/u0.txt')
        # u_init = u_init.reshape(trajopt.u.shape)
        # l_init = np.loadtxt('data/single_legged_hopper/nominal_3/l0.txt')
        # t_init = np.loadtxt('data/single_legged_hopper/nominal_3/t.txt')
        # R=  0.01*np.diag([1,1,1])
        # b = np.zeros((3,))
        # trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
        # Q = np.diag([1,10,10,100,100,1,1,1,1,1])
        # trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
    else: 
        # Add a running cost on the controls and state
        # add costs after the first iteration
        
        R=  0.01*np.diag([1,1,1])
        b = np.zeros((3,))
        trajopt.add_quadratic_running_cost(R, b, [trajopt.u], name="ControlCost")
        Q = np.diag([10,10,10,100,100,1,1,1,1,1])
        trajopt.add_quadratic_running_cost(Q, xf, [trajopt.x], name="StateCost")
        # load saved traj
        
        # num = i - 1
        # x_init = np.loadtxt('data/single_legged_hopper/nominal_3/x{n}.txt'.format(n = num))
        # u_init = np.loadtxt('data/single_legged_hopper/nominal_3/u{n}.txt'.format(n = num))
        # u_init = u_init.reshape(trajopt.u.shape)
        # l_init = np.loadtxt('data/single_legged_hopper/nominal_3/l{n}.txt'.format(n = num))
        # t_init = np.loadtxt('data/single_legged_hopper/nominal_3/t.txt')
        # jl_init = np.zeros(trajopt.jl.shape)
        dvars = trajopt.prog.decision_variables()
        vals = result.GetSolution(dvars)
        trajopt.prog.SetInitialGuess(dvars, vals)

        # u_init = result.GetSolution(trajopt.u)
        # x_init = result.GetSolution(trajopt.x)
        # l_init = result.GetSolution(trajopt.l)
        # jl_init = result.GetSolution(trajopt.jl)
        # s_init = result.GetSolution(trajopt.slacks)
        # trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj = jl_init, straj = s_init)
        
    # Get the final program, with all costs and constraints
    prog = trajopt.get_program()
    # Set the SNOPT solver options
    
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", tolerance)
    prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", tolerance)
    
    if i == 0:
        prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 1)
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
    u_init = result.GetSolution(trajopt.u)
    # Unpack and plot the trajectories
    x = trajopt.reconstruct_state_trajectory(result)
    u = trajopt.reconstruct_input_trajectory(result)
    l = trajopt.reconstruct_reaction_force_trajectory(result)
    t = trajopt.get_solution_times(result)
    # np.savetxt('data/single_legged_hopper/nominal_3/x{n}.txt'.format(n = i), x, fmt = '%1.3f')
    # np.savetxt('data/single_legged_hopper/nominal_3/u{n}.txt'.format(n = i), u, fmt = '%1.3f')
    # np.savetxt('data/single_legged_hopper/nominal_3/l{n}.txt'.format(n = i), l, fmt = '%1.3f')
    # np.savetxt('data/single_legged_hopper/nominal_3/t.txt'.format(n = i), t, fmt = '%1.3f')
    i += 1
x = trajopt.reconstruct_state_trajectory(result)
u = trajopt.reconstruct_input_trajectory(result)
l = trajopt.reconstruct_reaction_force_trajectory(result)
t = trajopt.get_solution_times(result)
jl= result.GetSolution(trajopt.jl)
s = result.GetSolution(trajopt.slacks)
save_num = 9
np.savetxt('data/single_legged_hopper/nominal_3/x_{n}.txt'.format(n = save_num), x, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/u_{n}.txt'.format(n = save_num), u, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/l_{n}.txt'.format(n = save_num), l, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/jl_{n}.txt'.format(n = save_num), jl, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/s_{n}.txt'.format(n = save_num), s, fmt = '%1.3f')
np.savetxt('data/single_legged_hopper/nominal_3/t.txt', t, fmt = '%1.3f')
plot(x,u, l, t) 
x = PiecewisePolynomial.FirstOrderHold(t, x)
vis = Visualizer(_file)
body_inds = vis.plant.GetBodyIndices(vis.model_index)
base_frame = vis.plant.get_body(body_inds[0]).body_frame()
vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())

vis.visualize_trajectory(x)

print('Done!')