import os, errno
from sys import exit
from pydrake.autodiffutils import AutoDiffXd
from matplotlib import pyplot as plt
import pickle
import numpy as np
import re

SNOPT_DECODER = {
    0: "finished successfully",
    1: "optimality conditions satisfied",
    2: "feasible point found",
    3: "requested accuracy could not be achieved",
    11: "infeasible linear constraints",
    12: "infeasible linear equalities",
    13: "nonlinear infeasibilities minimized",
    14: "infeasibilities minimized",
    21: "unbounded objective",
    22: "constraint violation limit reached",
    31: "iteration limit reached",
    32: "major iteration limit reached",
    33: "the superbasics limit is too small",
    41: "current point cannot be improved",
    42: "singular basis",
    43: "cannot satisfy the general constraints",
    44: "ill-conditioned null-space basis",
    51: "incoorrect objective derivatives",
    52: "incorrect constraint derivatives",
    61: "undefined function at the first feasible point",
    62: "undefined function at the initial point",
    63: "unable to proceed in undefined region",
    71: "terminated during function evaluation",
    72: "terminated during constraint evaluation",
    73: "terminated during objective evaluation",
    74: "termianted from monitor routine",
    81: "work arrays must have at least 500 elements",
    82: "not enough character storage",
    83: "not enough integer storage",
    84: "not enough real storage",
    91: "invalid input argument",
    92: "basis file dimensions do not match this problem",
    141: "wrong number of basic variables",
    142: "error in basis package"
}
#TODO: Add "save_and_close" to MathProgIterationPrinter for saving cost figure
class MathProgIterationPrinter():
    def __init__(self, prog = None, display='terminal'):
        """
            display options:
                "terminal" prints costs and constraints into the terminal
                "figure" creates a matplotlib window and plots all costs and constraints in subplots
                "all" prints to the terminal and creates a figure
        """
        self._prog = prog
        self.iteration = 0
        self._thresh = 1e-6
        self.fig = None
        self.display_func = self._get_display_func(display)
        self.title_iter = 50 #Print titles to terminal every title_iter iterations
        
    def __call__(self, x):
        costs = self.calc_costs(x)
        cstrs = self.calc_constraints(x)
        self.iteration += 1
        self.display_func(costs, cstrs)

    def _get_display_func(self, display):
        """ Returns the appropriate display method """
        if display.lower() == 'terminal':
            return self.print_to_terminal
        elif display.lower() == 'figure':
            self.figure_setup()
            return self.print_to_figure
        elif display.lower() == 'all':
            self.figure_setup()
            return self.print_to_terminal_and_figure
        else:
            raise ValueError(f"Display {display} is not a supported option. Choose 'terminal', 'figure', or 'all'")

    def reset(self):
        """
        Reset the iteration printer.
            If the previous figure hasn't been closed, close it
            Create a new figure if necessary
            Reset the iteration counter to zero
        """
        # Reset iteration counter
        self.iteration = 0
        if self.fig is not None:
            self.figure_setup()

    def calc_costs(self, x):
        """ 
        Calculate and return all cost function values from the mathematical program
        
        Arguments:
            x: array of all program decision variables, in order.
        """
        costs = self._prog.GetAllCosts()
        cost_vals = {}
        for cost in costs:
            dvars = cost.variables()
            # Filter out the necessary variables
            dvals = x[self._prog.FindDecisionVariableIndices(dvars)]
            # Evaluate the cost
            val = cost.evaluator().Eval(dvals)
            # Add to the cost dictionary
            name = cost.evaluator().get_description()
            if name in cost_vals:
                cost_vals[name] += val[0]   # Unwrap the array
            else:
                cost_vals[name] = val[0]
        return cost_vals

    def calc_constraints(self, x):
        """ 
        Calculate and return all constraint violations from the mathematical program
        
        Arguments:
            x: array of all program decision variables, in order
        """
        cstrs = self._prog.GetAllConstraints()
        cstr_vals = {}
        for cstr in cstrs:
            dvars = cstr.variables()
            # Filter out necessary variables
            dvals = x[self._prog.FindDecisionVariableIndices(dvars)]
            # Evaluate the cost
            val = cstr.evaluator().Eval(dvals)
            # Get absolute constraint violations
            lb = cstr.evaluator().lower_bound()
            ub = cstr.evaluator().upper_bound()
            lb_viol = np.minimum(val - lb, np.zeros((lb.shape)))
            lb_viol[np.isinf(lb)] = 0.
            ub_viol = np.maximum(val - ub, np.zeros((ub.shape)))
            ub_viol[np.isinf(ub)] = 0.
            viol = sum(abs(lb_viol)) + sum(abs(ub_viol))
            # Add to constraint dictionary
            name = cstr.evaluator().get_description()
            if name in cstr_vals:
                cstr_vals[name] += viol
            else:
                cstr_vals[name] = viol
        return cstr_vals

    def print_to_terminal(self, costs, cstrs):
        """ Print costs and constraints to the terminal"""

        # Create the title and value strings
        title_str = '{1:<{0}s}'.format(15, 'Iteration')
        value_str = '{1:<{0}f}'.format(15, self.iteration)
        for name, value in costs.items():
            #  Make sure the names and values line up and have spaces between them
            width = max(len(name) + 5, 15)
            title_str += '{1:<{0}s}'.format(width, name)
            value_str += '{1:<{0}.8E}'.format(width, value)

        for name, value in cstrs.items():
            width = max(len(name) + 5, 15)
            title_str += '{1:<{0}s}'.format(width, name)
            value_str += '{1:<{0}.8E}'.format(width, value)

        # Print the names of the costs and constraints
        if self.iteration % self.title_iter == 1:
            print(title_str)
        # Print the values of the costs and constraints
        print(value_str)

    def print_to_figure(self, costs, cstrs):
        """ Print costs and constraints to a figure window"""
        # kEps = np.finfo(float).eps
        # kExp = int(np.log10(kEps)) + 2  # Set the floor to 100 times machine precision
        # Note: Initialize the lines
        if self.iteration == 1:
            for name, value in costs.items():
                self.cost_lines[name] = self.axs[0].plot([self.iteration], [value], linewidth=1.5, label=name)[0]
            for name, value in cstrs.items():
                self.cstr_lines[name] = self.axs[1].plot([self.iteration], [value], linewidth=1.5, label=name)[0]
            self.axs[0].legend()
            self.axs[1].legend()
        else:
            ymax = self.axs[0].get_ylim()[1]
            for name, value in costs.items():
                x = np.append(self.cost_lines[name].get_xdata(), self.iteration)
                y = np.append(self.cost_lines[name].get_ydata(), value)
                self.cost_lines[name].set_data((x, y))
                ymax = max(value, ymax)
            #Set new axis limits
            self.axs[0].set_xlim([1, self.iteration])
            self.axs[0].set_ylim([0, ymax])

            ymax = self.axs[1].get_ylim()[1]
            for name, value in cstrs.items():
                x = np.append(self.cstr_lines[name].get_xdata(), self.iteration)
                y = np.append(self.cstr_lines[name].get_ydata(), value)
                self.cstr_lines[name].set_data((x, y))
                ymax = max(value, ymax)
            # Set new axis limits
            self.axs[1].set_xlim([1,self.iteration])
            self.axs[1].set_ylim([0, ymax])
        # Draw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Pause to let the figure show
        plt.pause(0.01)

    def figure_setup(self):
        self.fig, self.axs = plt.subplots(2,1)
        self.axs[0].set_ylabel('Cost')
        self.axs[0].set_yscale('symlog', linthreshy=self._thresh)
        self.axs[0].grid(True)
        self.axs[1].set_ylabel('Constraint Violation')
        self.axs[1].set_xlabel('Iteration')
        self.axs[1].set_yscale('symlog', linthreshy=self._thresh)
        self.axs[1].grid(True)
        self.cost_lines = {}
        self.cstr_lines = {}

    def print_to_terminal_and_figure(self, costs, cstrs):
        self.print_to_terminal(costs, cstrs)
        self.print_to_figure(costs, cstrs)

    def save_and_close(self, savename="CostsAndConstraints.png"):
        """Save and close the figure created by MathematicalProgram's VisualizeCallback function"""
        if self.fig is not None:
            self.fig.savefig(savename, dpi = self.fig.dpi)
            plt.close(self.fig)

    def save_and_clear(self, savename="CostsAndConstraints.png"):
        if self.fig is not None:
            self.fig.savefig(savename, dpi=self.fig.dpi)
            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[0].set_ylabel('Cost')
            self.axs[0].set_yscale('symlog', linthreshy=self._thresh)
            self.axs[0].grid(True)
            self.axs[1].set_ylabel('Constraint Violation')
            self.axs[1].set_xlabel('Iteration')
            self.axs[1].set_yscale('symlog', linthreshy=self._thresh)
            self.axs[1].grid(True)
            self.iteration = 0

    @property
    def title_iter(self):
        return self._title_iter
    
    @title_iter.setter
    def title_iter(self, val):
        if type(val) is int and val > 0:
            self._title_iter = val
        else:
            raise ValueError(f"title_iter must be a nonnegative integer")

def append_filename(name, append_str):
    if name is None:
        return None
    else:
        parts = name.split(".")
        parts[0] += append_str
        return ".".join(parts)

def save(filename, data):
    """ pickle data in the specified filename """
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(filename, "wb") as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

def load(filename):
    """ unpickle the data in the specified filename """
    with open(filename, "rb") as input:
        data = pickle.load(input)
    return data

def FindResource(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    else:
        return os.path.abspath(filename)
    
def CheckProgram(prog):
    """
    Return true if the outputs of all costs and constraints in MathematicalProgram are valid
    
    Arguments:
        prog: a MathematicalProgram pyDrake object
    """
    status = True
    # Check that the outputs of the costs are all scalars
    for cost in prog.generic_costs():
        # Evaluate the cost with floats
        try:
            xs = [1.]*len(cost.variables())
            cost.evaluator().Eval(xs)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cost.evaluator().get_description()} with floats produces a RuntimeError")
        # Evaluate with AutoDiff arrays
        try:
            xd = [AutoDiffXd(1.)] * len(cost.variables())
            cost.evaluator().Eval(xd)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cost.evaluator().get_description()} with AutoDiffs produces a RuntimeError")
    # Check that the outputs of all constraints are vectors
    for cstr in prog.generic_constraints():
        # Evaluate the constraint with floats
        try:
            xs = [1.]*len(cstr.variables())
            cstr.evaluator().Eval(xs)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with floats produces a RuntimeError")
        except ValueError as err:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with floats resulted in a ValueError")
        # Evaluate constraint with AutoDiffXd
        try:
            xd = [AutoDiffXd(1.)] * len(cstr.variables())
            cstr.evaluator().Eval(xd)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with AutoDiffs produces a RuntimeError")
        except ValueError as err:
            print(f"Evaluating {cstr.evaluator().get_description()} with AutoDiffs produces a ValueError")
    # Return the status flag
    return status

def GetKnotsFromTrajectory(trajectory):
    breaks = trajectory.get_segment_times()
    values = trajectory.vector_values(breaks)
    return (breaks, values)

def printProgramReport(result, prog=None, terminal=True, filename=None, verbose=False):
    """print out information about the result of the mathematical program """
    # Print out general information
    report = f"Solved with {result.get_solver_id().name()}\n"
    report += f"Optimization successful? {result.is_success()}\n"
    report += f"Optimal cost = {result.get_optimal_cost()}\n"
    # Print out SNOPT specific information
    if result.get_solver_id().name() == "SNOPT/fortran":
        exit_code = result.get_solver_details().info
        report += f"SNOPT Exit Status {exit_code}: {SNOPT_DECODER[exit_code]}\n"
        if prog is not None:
            # Filter out the empty infeasible constraints
            infeasibles = result.GetInfeasibleConstraintNames(prog)
            infeas = [name.split("[")[0] for name in infeasibles]
            report += f"Infeasible constraints: {set(infeas)}\n"
    # Print out verbose cost and constraint information
    if verbose:
        printer = MathProgIterationPrinter(prog)
        all_vars = result.GetSolution(prog.decision_variables())
        costs = printer.calc_costs(all_vars)
        cstrs = printer.calc_constraints(all_vars)
        report += "Individual Costs: \n"
        for key in costs:
            report += f"{key}: \t {costs[key]:.4E}\n"
        report += "\nConstraint Violations: \n"
        for key in cstrs:
            report += f"{key}: \t {cstrs[key]:.4E}\n"
    # Print the report to terminal
    if terminal:
        print(report)
    #Save to file 
    if filename is not None:
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(filename, "w") as file:
            file.write(report)
    #Return the report as a text string
    return report

def quat2rpy(quat):
    """
    Convert a quaternion to Roll-Pitch-Yaw
    
    Arguments:
        quaternion: a (4,n) numpy array of quaternions
    
    Return values:
        rpy: a (3,n) numpy array of roll-pitch-yaw values
    """
    rpy = np.zeros((3, quat.shape[1]))
    rpy[0,:] = np.arctan2(2*(quat[0,:]*quat[1,:] + quat[2,:]*quat[3,:]),
                         1-2*(quat[1,:]**2 + quat[2,:]**2))
    rpy[1,:] = np.arcsin(2*(quat[0,:]*quat[2,:]-quat[3,:]*quat[1,:]))
    rpy[2,:] = np.arctan2(2*(quat[0,:]*quat[3,:]+quat[1,:]*quat[2,:]),
                        1-2*(quat[2,:]**2 + quat[3,:]**2))
    return rpy

def plot_complementarity(ax, x, y1, y2, label1, label2):
    """
        Plots two traces in the same axes using different y-axes. Aligns the y-axes at zero

        Arguments:
            ax: The axis on which to plot
            y1: The first sequence to plot
            y2: The second sequence to plot
            label1: The y-axis label for the first sequence, y1
            label2: The y-axis label for the second sequence, y2
    """
    if x is None:
        x = range(0, len(y1))
    color = "tab:red"
    ax.set_ylabel(label1, color = color)
    ax.plot(x, y1, "-", color=color, linewidth=1.5)
    # Create the second axis 
    ax2 = ax.twinx()
    color = "tab:blue"
    ax2.set_ylabel(label2, color=color)
    ax2.plot(x, y2, "-", color=color, linewidth=1.5)
    # Align the axes at zero
    align_axes(ax,ax2)

def align_axes(ax, ax2):
    """
        For a plot with two y-axes, aligns the two y-axes at 0

        Arguments:
            ax: Reference to the first of the two y-axes
            ax2: Reference to the second of the two y-axes
    """
    lims = np.array([ax.get_ylim(), ax2.get_ylim()])
    # Pad the limits to make sure there is some range
    lims += np.array([[-1,1],[-1,1]])
    lim_range = lims[:,1] - lims[:,0]
    lim_frac = lims.transpose() / lim_range
    lim_frac = lim_frac.transpose()
    new_frac = np.array([min(lim_frac[:,0]), max(lim_frac[:,1])])
    ax.set_ylim(lim_range[0]*new_frac)
    ax2.set_ylim(lim_range[1]*new_frac)

def getDualSolutionDict(prog, result):
    """ Returns the dual solutions of all constraints in a dictionary"""
    duals = {}
    # Get the dual solutions and add them to a dictionary
    for cstr in prog.GetAllConstraints():
        name = cstr.evaluator().get_description()
        dual = result.GetDualSolution(cstr)
        if name in duals:
            duals[name].append(dual)
        else:
            duals[name] = [dual]
    # Stack all repeating duals along the rows
    for name in duals.keys():
        duals[name] = np.row_stack(duals[name])
    return duals

def alphanumeric_sort(text_list):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(text_list, key=alphanum_key)

def find_filepath_recursive(directory, target_file):
    for path, dir, files in os.walk(directory):
        for file in files:
            if file == "trajoptresults.pkl":
                yield path
