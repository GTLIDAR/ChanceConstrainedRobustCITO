import os
from sys import exit
from pydrake.autodiffutils import AutoDiffXd
import pickle

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
        exit(f"{filename} not found")
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
        except RuntimeError:
            status = False
            print(f"Evaluating {cost.evaluator().get_description()} with floats produces a RuntimeError")
        # Evaluate with AutoDiff arrays
        try:
            xd = [AutoDiffXd(1.)] * len(cost.variables())
            cost.evaluator().Eval(xd)
        except RuntimeError:
            status = False
            print(f"Evaluating {cost.evaluator().get_description()} with AutoDiffs produces a RuntimeError")

    # Check that the outputs of all constraints are vectors
    for cstr in prog.generic_constraints():
        # Evaluate the constraint with floats
        try:
            xs = [1.]*len(cstr.variables())
            cstr.evaluator().Eval(xs)
        except RuntimeError:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with floats produces a RuntimeError")
        # Evaluate constraint with AutoDiffXd
        try:
            xd = [AutoDiffXd(1.)] * len(cstr.variables())
            cstr.evaluator().Eval(xd)
        except RuntimeError:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with AutoDiffs produces a RuntimeError")
    # Return the status flag
    return status