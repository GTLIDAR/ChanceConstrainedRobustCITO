"""
Helpful tools for analyzing a solution from trajectory optimization

Luke Drnach
September 3, 2021
"""
import numpy as np
import decorators as deco 
import utilities as utils
import matplotlib.pyplot as plt
import os

class MeritCalculator():
    def __init__(self, trajopt):
        # Store a copy of the trajectory optimization
        self.trajopt = trajopt

    def merit_scores(self, result_dict, input_merit_trajectory=False):
        """Return the merit score, the mean-squared constraint violations"""
        if not input_merit_trajectory:
            merit_traj = self.merit_trajectory(result_dict)
        else:
            merit_traj = result_dict
        merit_score = {}
        for key, value in merit_traj.items():
            merit_score[key] = np.average(value)
        return merit_score

    def merit_trajectory(self, result_dict):
        """Return the merit score for each timepoint as a trajectory"""
        return self.calc_constraint_violation(result_dict)

    def calc_constraint_violation(self, result_dict):
        """Return a dictionary of all constraint values"""
        # First get the values of the decision variables
        allvals = self.dict_to_dvals(result_dict)
        all_cstr = {}
        for cstr in self.trajopt.prog.GetAllConstraints():
            # Get the variables for the constraint
            dvars = cstr.variables()
            dvals = allvals[self.trajopt.prog.FindDecisionVariableIndices(dvars)]
            # Evaluate the constraint
            cval = cstr.evaluator().Eval(dvals)
            # Get the absolute constraint violation
            lb = cstr.evaluator().lower_bound()
            ub = cstr.evaluator().upper_bound()
            lb_viol = np.minimum(cval - lb, np.zeros((lb.shape)))
            ub_viol = np.maximum(cval - ub, np.zeros((ub.shape)))
            cviol = np.maximum(np.abs(lb_viol), np.abs(ub_viol))**2
            # Sort by constraint name
            cname = cstr.evaluator().get_description()
            if cname in all_cstr.keys():
                all_cstr[cname].append(np.sum(cviol))
            else:
                all_cstr[cname] = [np.sum(cviol)]
        # Convert each of the constraint violations into a numpy array
        for key in all_cstr.keys():
            all_cstr[key] = np.squeeze(np.vstack(all_cstr[key]))
        return all_cstr

    def dict_to_dvals(self, result_dict):
        dvals = np.zeros((self.trajopt.prog.num_vars(),))
        Finder = self.trajopt.prog.FindDecisionVariableIndices
        # Add in time variables
        dvals[Finder(self.trajopt.h.flatten())] = np.diff(result_dict['time']).flatten()
        # Add in state variables
        dvals[Finder(self.trajopt.x.flatten())] = result_dict['state'].flatten()
        # Add in control variables
        dvals[Finder(self.trajopt.u.flatten())] = result_dict['control'].flatten()
        # Add in force variables
        dvals[Finder(self.trajopt.l.flatten())] = result_dict['force'].flatten()
        # Add in limit variables (if any)
        if 'jointlimit' in result_dict and result_dict['jointlimit'] is not None:
            dvals[Finder(self.trajopt.jl.flatten())] = result_dict['jointlimit'].flatten()
        # Add in slack variables (if any)
        if 'slacks' in result_dict and result_dict['slacks'] is not None and self.trajopt.slacks is not None:
            dvals[Finder(self.trajopt.slacks.flatten())] = result_dict['slacks'].flatten()
        return dvals

    def friction_sensitivity(self, result_dict, stepsize = 1e-4):
        """
        Calculate sensitivity of the merit score to changes in friction
        Works only for terrain with constant friction
        """
        friction = self.trajopt.plant_f.terrain.friction
        # Perturb friction in the positive direction
        self.trajopt.plant_f.terrain.friction = friction + stepsize
        trajectory_plus = self.merit_trajectory(result_dict)
        merit_plus = self.merit_scores(result_dict)
        # Perturb friction in the negative direction
        self.trajopt.plant_f.terrain.friction = friction - stepsize
        trajectory_minus = self.merit_trajectory(result_dict)
        merit_minus = self.merit_scores(result_dict)
        # Reset friction
        self.trajopt.plant_f.terrain.friction = friction
        # Calculate the sensitivity - the change in merit
        sensitivity_trajectory = {}
        sensitivity_score = {}
        for key in trajectory_plus.keys():
            sensitivity_trajectory[key] = (trajectory_plus[key] - trajectory_minus[key])/(2*stepsize)
            sensitivity_score[key] = (merit_plus[key] - merit_minus[key])/(2*stepsize)
        return sensitivity_score, sensitivity_trajectory

    def height_sensitivity(self, result_dict, stepsize = 1e-4):
        """
        Calculate sensitivity of the merit score to changes in terrain height
        Works only for terrain with constant height
        """
        height = self.trajopt.plant_f.terrain.height
        # Perturb friction in the positive direction
        self.trajopt.plant_f.terrain.height = height + stepsize
        trajectory_plus = self.merit_trajectory(result_dict)
        merit_plus = self.merit_scores(result_dict)
        # Perturb friction in the negative direction
        self.trajopt.plant_f.terrain.height = height - stepsize
        trajectory_minus = self.merit_trajectory(result_dict)
        merit_minus = self.merit_scores(result_dict)
        # Reset friction
        self.trajopt.plant_f.terrain.height = height
        # Calculate the sensitivity - the change in merit
        sensitivity_trajectory = {}
        sensitivity_score = {}
        for key in trajectory_plus.keys():
            sensitivity_trajectory[key] = (trajectory_plus[key] - trajectory_minus[key])/(2*stepsize)
            sensitivity_score[key] = (merit_plus[key] - merit_minus[key])/(2*stepsize)

        return sensitivity_score, sensitivity_trajectory

    @deco.showable_fig
    @deco.saveable_fig
    def plot_merit_scores(self, scores):
        """Calculate and plot the merit scores in a bar chart""" 
        names = list(scores.keys())
        values = [scores[key] for key in names]
        x = [n for n in range(len(values))]
        fig, axs = plt.subplots(1,1)
        axs.bar(x, values)
        plt.xticks(x, names, rotation=20)
        axs.set_title("Mean-Squared Infeasibility")
        axs.set_xlabel("Constraint")
        axs.set_ylabel("Merit Score")
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_merit_trajectory(self, trajectories):
        """Plot the merit scores as trajectories"""
        keys = ['dynamics','normal_distance','sliding_velocity','friction_cone']
        if 'joint_limits' in trajectories.keys():
            keys.append('joint_limits')
        fig, axs = plt.subplots(1,1)
        for key in keys:
            t = np.linspace(0, 1, trajectories[key].shape[0])
            axs.plot(t, trajectories[key], linewidth=1.5, label=key)
        axs.legend()
        axs.set_ylabel('Merit Score')
        axs.set_xlabel('Normalized time')
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_sensitivities(self, friction, height):
        """Plot the friction and terrain height sensitivities"""
        # Calculate the data
        fscore, ftrajectory = friction
        hscore, htrajectory = height
        # Generate the plot
        fig, axs = plt.subplots(3,1)
        t = np.linspace(0, 1, ftrajectory['friction_cone'].shape[0])
        axs[0].plot(t, ftrajectory['friction_cone'], linewidth=1.5)
        axs[0].set_ylabel('Friction Sensitivity')
        axs[1].plot(t, htrajectory['normal_distance'], linewidth=1.5)
        axs[1].set_ylabel('Height Sensitivity')
        axs[1].set_xlabel('Normalized Time')
        axs[2].bar(['Friction','Height'],[fscore['friction_cone'],hscore['normal_distance']])
        axs[2].set_ylabel('Mean Sensitivity')
        return fig, axs

class ContactResidualCalculator(MeritCalculator):
    def __init__(self, trajopt):
        super(ContactResidualCalculator, self).__init__(trajopt)
        self.residual_fcn = min_residual
    
    def useMinResidual(self):
        self.residual_fcn = min_residual

    def useFisherResidual(self):
        self.residual_fcn = fisher_residual

    def calc_constraint_violation(self, result_dict):
        # Get the states and forces from the dictionary
        states = result_dict['state']
        forces = result_dict['force']
        # Important indices
        numN = self.trajopt.numN
        numT = self.trajopt.numT
        # Initialize the arrays
        residuals = {'normal_distance': np.zeros((numN, states.shape[1])),
                    'sliding_velocity': np.zeros((numT, states.shape[1])),
                    'friction_cone': np.zeros((numN, states.shape[1]))
        }
        for n in range(states.shape[1]):
            # Calculate the complementarity functions
            nd = self.trajopt._normal_distance(states[:,n])
            sv = self.trajopt._sliding_velocity(np.concatenate([states[:,n], forces[numN+numT:, n]], axis=0))
            fc = self.trajopt._friction_cone(np.concatenate([states[:,n], forces[:numN+numT, n]], axis=0))
            # Calculate the residuals
            residuals['normal_distance'][:,n] = self.residual_fcn(nd, forces[:numN, n])
            residuals['sliding_velocity'][:,n] = self.residual_fcn(sv, forces[numN:numN+numT,n])
            residuals['friction_cone'][:,n] = self.residual_fcn(fc, forces[numN+numT:, n])
        # Sum squared residuals
        for key in residuals.keys():
            residuals[key] = np.sum(residuals[key]**2, axis=0)
        
        return residuals

    @deco.showable_fig
    @deco.saveable_fig
    def plot_merit_trajectory(self, trajectories):
        """Plot the merit scores as trajectories"""
        keys = ['normal_distance','sliding_velocity','friction_cone']
        fig, axs = plt.subplots(1,1)
        for key in keys:
            t = np.linspace(0, 1, trajectories[key].shape[0])
            axs.plot(t, trajectories[key], linewidth=1.5, label=key)
        axs.legend()
        axs.set_ylabel('Merit Score')
        axs.set_xlabel('Normalized time')
        return fig, axs

def min_residual(a, b):
    return np.minimum(a,b)

def fisher_residual(a, b):
    fb = a + b - np.sqrt(a**2 + b**2)
    return fb

def run_merit_analysis(calculator, filepath, filename='trajoptresults.pkl'):
    fullfile = os.path.join(filepath, filename)
    print(f"Calculating merit for {fullfile}")
    # Append merit scores to the data
    data = utils.load(fullfile)
    data['merit_traj'] = calculator.merit_trajectory(data)
    data['merit_score'] = calculator.merit_scores(data)
    # Add in sensitivity analysis
    fricScore, fricTraj = calculator.friction_sensitivity(data)
    heightScore, heightTraj = calculator.height_sensitivity(data)
    data['sensitivity_score'] = {'friction_cone': fricScore,
                                'normal_distance': heightScore}
    data['sensitivity_traj'] = {'friction_cone': fricTraj,
                                'normal_distance': heightTraj}
    # Make plots of the merit trajectory
    print(f"Plotting merit trajectories")
    fig1, _ = calculator.plot_merit_trajectory(data['merit_traj'], show=False, savename=os.path.join(filepath, 'merit_trajectory.png'))
    fig2, _ = calculator.plot_merit_scores(data['merit_score'], show=False, savename=os.path.join(filepath, "merit_scores.png"))
    # Make plots of the sensitivity
    print(f"Plotting sensitivities")
    fig3, _ = calculator.plot_sensitivities((fricScore, fricTraj), (heightScore, heightTraj), show=False, savename=os.path.join(filepath,'sensitivity.png'))
    # Resave the data
    print(f"Resaving the data")
    utils.save(fullfile, data)
    # Close any opened plots
    for fig in [fig1, fig2, fig3]:
        plt.close(fig)

def batch_run_merit_analysis_fisher(trajopt, directory, filename='trajoptresults.pkl'):
    meritCalc = ContactResidualCalculator(trajopt)
    meritCalc.useFisherResidual()
    for filepath in utils.find_filepath_recursive(directory, filename):
        run_merit_analysis(meritCalc, filepath, filename)

def batch_run_merit_analysis_min(trajopt, directory, filename='trajoptresults.pkl'):
    meritCalc = ContactResidualCalculator(trajopt)
    meritCalc.useMinResidual()
    for filepath in utils.find_filepath_recursive(directory, filename):
        run_merit_analysis(meritCalc, filepath, filename)

def batch_run_merit_analysis(trajopt, directory, filename='trajoptresults.pkl'):
    meritCalc = MeritCalculator(trajopt)
    for filepath in utils.find_filepath_recursive(directory, filename):
        run_merit_analysis(meritCalc, filepath, filename)
        
