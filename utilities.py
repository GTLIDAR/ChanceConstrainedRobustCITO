from os import path 
from sys import exit
from numpy import isscalar

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
        # Evaluate the cost
        xs = [1]*len(cost.variables())
        out = cost.evaluator().Eval(xs)
        if not isscalar(out):
            print(f"{cost.evaluator().get_description()} returns a vector instead of a scalar")
            status = False
    # Check that the outputs of all constraints are vectors
    for cstr in prog.generic_constraints():
        # Evaluate the constraint
        xs = [1]*len(cstr.variables())
        out = cstr.evaluator().Eval(xs)
        if isscalar(out):
            print(f"{cstr.evaluator().get_description()} returns a scalar instead of a vector")
            status = False
    # Return the status flag
    return status