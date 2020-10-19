from os import path 
from sys import exit
from pydrake.autodiffutils import AutoDiffXd

def FindResource(filename):
    if not path.isfile(filename):
        exit(f"{filename} not found")
    else:
        return path.abspath(filename)
    
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