from pydrake.all import MathematicalProgram, Solve
import numpy as np

M = np.array([[2,1],[0,2]])
q = np.array([-1,-2])

prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddLinearComplementarityConstraint(M, q, x)

# Solve the complementarity problem
result = Solve(prog)
# Print the result
print(f"Success: {result.is_success()}")
print(f"x = {result.GetSolution(x)}")
print(f"solver is: {result.get_solver_id().name()}")