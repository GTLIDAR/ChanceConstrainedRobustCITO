import numpy as np
from systems.hopper.hopper import Hopper
import trajopt.contactimplicit as ci
import os 
import utilities as utils
import matplotlib.pyplot as plt

def create_hopper():
    """Create a footed hopper model"""
    hopper = Hopper()
    hopper.Finalize()
    return hopper

def boundary_conditions(hopper):
    """Generate boundary conditions for optimization"""
    # Create boundary constraints
    base_0 = np.array([0., 1.5])
    base_f = np.array([4., 1.5])

    q_0, status = hopper.standing_pose_ik(base_0)
    if not status:
        print("Standing pose IK for Hopper not solved successfully. Returning last iteration")
    #q_0 = hopper.standing_pose(base_0)
    no_vel = np.zeros((5,))
    x_0 = np.concatenate((q_0, no_vel), axis=0)
    x_f = x_0.copy()
    x_f[:2] = base_f[:]
    return x_0, x_f

def create_hopper_optimization_contact_cost(hopper, x_0, x_f, N=101):
    context = hopper.multibody.CreateDefaultContext()
    options = ci.OptimizationOptions()
    options.useNonlinearComplementarityWithCost()
    # Create the optimization
    max_time = 3
    min_time = 3
    trajopt = ci.ContactImplicitDirectTranscription(hopper, context, num_time_samples=N, minimum_timestep=min_time/(N-1), maximum_timestep=max_time/(N-1), options=options)
    # Add boundary constraints
    trajopt.add_state_constraint(knotpoint=0, value=x_0)
    trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=x_f)

    trajopt.setSolverOptions({'Iterations limit': 100000,
                            'Major iterations limit': 5000,
                            'Minor iterations limit': 1000, 
                            'Superbasics limit': 1500,
                            'Scale option': 1,
                            'Elastic weight': 10**5})
    trajopt.enable_cost_display('figure')
    # Require equal timesteps
    trajopt.add_equal_time_constraints()
    # Set the force scaling
    trajopt.force_scaling = 1
    # Add in running cost
    R = 0.01*np.eye(3)
    Q = np.diag([1, 10, 10, 100, 100, 1, 1, 1, 1, 1])
    R = R/2
    Q = Q/2

    trajopt.add_quadratic_running_cost(R, np.zeros((3,)), vars=[trajopt.u], name='ControlCost')
    trajopt.add_quadratic_running_cost(Q, x_f, vars=[trajopt.x], name='StateCost')
    return trajopt

def solve_hopper_opt_and_save(trajopt, savedir):
    # Check if the save directory exists
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    else:
        ans = input(f"{savedir} already exists. Would you like to overwrite it? (Y/n)>")
        if ans.lower() != 'y':
            return None
    # Solve the problem
    result = trajopt.solve()
    print(f"Successful? {result.is_success()}")
    # Save the outputs
    report = trajopt.generate_report(result)
    reportfile = os.path.join(savedir, 'report.txt')
    with open(reportfile, 'w') as file:
        file.write(report)
    utils.save(os.path.join(savedir, 'trajoptresults.pkl'), trajopt.result_to_dict(result))
    # Save the cost figure
    trajopt.printer.save_and_clear(os.path.join(savedir, 'CostsAndConstraints.png'))
    # Plot and save the trajectories
    xtraj, utraj, ftraj, jltraj, _ = trajopt.reconstruct_all_trajectories(result)
    figs, _ = trajopt.plant_f.plot_trajectories(xtraj, utraj, ftraj, jltraj, show=False, savename=os.path.join(savedir, 'opt.png'))
    for fig in figs:
        plt.close(fig)
    return result

def set_hopper_initial_conditions(trajopt, result=None, boundary=None):
    if result is None:
        dvars = trajopt.prog.decision_variables()
        dvals = np.zeros(dvars.shape)
        trajopt.prog.SetInitialGuess(dvars, dvals)
        trajopt._set_initial_timesteps()
        if boundary is not None:
            x0, xf = boundary
            xtraj = np.linspace(x0, xf, trajopt.num_time_samples).transpose()
            trajopt.prog.SetInitialGuess(trajopt.x, xtraj)
    else:
        trajopt.initialize_from_previous(result)
    return trajopt