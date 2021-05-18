"""
constraints: a package of extra constraints for the pyCITO project
These constraints are NOT subclasses of Drake's Constraint class
"""
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import erfinv
class NCCSlackType(Enum):
    """
    NCCSlackType controls the implementation of constraint relaxation for NonlinearComplementarityFcn
    NCCSlackType is an enumerated type. The following options are available.
    
    CONSTANT_SLACK: Complementarity with a fixed value for slack in the product constraint is implemented. To recover strict complementarity, set the slack to 0 
    
    VARIABLE_SLACK: Complementarity where the slack in the produce constraint is a decision variable. In this case, an additional cost should be added to penalize the slack variable
    """
    CONSTANT_SLACK = 1
    VARIABLE_SLACK = 3

class NCCImplementation(Enum):
    """
        NCCImplementation is an enumerated type controlling how NonlinearComplementarityFcn implements the constraint:
            f(z) >= 0
            g(z) >= 0
            f(g)*g(z) <= 0

        NONLINEAR: The constraint is implemented in its original format

        LINEAR_EQUALITY: Additional variables are passed to the constraint to implement a linear complementarity condition, and equality constraints are added. The new constraint set is:
            a = f(z)
            b = g(z)
            a >= 0
            b >= 0
            a*b <= 0

        COST: Only the non-negativity constraints are implemented as constraints. The product constraint must be added separately as a cost, using the method "product_cost". When using COST, the slack variable is ignored, but a cost weight is included to enforce the complementarity constraint
    """
    NONLINEAR = 1
    LINEAR_EQUALITY = 2
    COST = 3

class ComplementarityFactory():
    """
        ComplementarityFactory: Factory class for creating concrete implementations of nonlinear complementarity constraints for trajectory optimization or other mathematical programs.
    """
    def __init__(self, implementation=NCCImplementation.NONLINEAR, slacktype=NCCSlackType.CONSTANT_SLACK):
        self._constraint_class = self._get_constraint_class(implementation, slacktype)

    def create(self, fcn, xdim, zdim):
        """ 
        Create a concrete instance of the complementarity constraint
        """
        return self._constraint_class(fcn, xdim, zdim)

    @staticmethod
    def _get_constraint_class(implementation, slacktype):
        """
        Determine and return a class reference to make complementarity constraints with the specified implementation (determined by NCCImplementation type) and slack variable (determined by NCCSlackType)
        """
        if implementation == NCCImplementation.NONLINEAR:
            if slacktype == NCCSlackType.CONSTANT_SLACK:
                return ConstantSlackNonlinearComplementarity
            elif slacktype == NCCSlackType.VARIABLE_SLACK:
                return VariableSlackNonlinearComplementarity
            else:
                raise ValueError(f"Unknown Complementarity Slack Implementation {slacktype}")
        elif implementation == NCCImplementation.LINEAR_EQUALITY:
            if slacktype == NCCSlackType.CONSTANT_SLACK:
                return ConstantSlackLinearEqualityComplementarity
            elif slacktype == NCCSlackType.VARIABLE_SLACK:
                return VariableSlackLinearEqualityComplementarity
            else:
                raise ValueError(f"Unknown Complementarity Slack Implementation {slacktype}")
        elif implementation == NCCImplementation.COST:
            return CostRelaxedNonlinearComplementarity
        else:
            raise ValueError(f"Unknown Complementarity Function Implementation {implementation}")

class ComplementarityFunction(ABC):
    """
    Base class for implementing complementarity constraint functions of the form:
        f(x) >= 0
        z >= 0
        f(x) * z <= s
    where s is a slack variable and s=0 enforces strict complementarity
    """
    def __init__(self, fcn, xdim=0, zdim=1):
        """
            Creates the complementarity constraint function. Stores a reference to the nonlinear function, the dimensions of the x and z variables, and sets the slack to 0
        """
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.slack = 0.
    
    def __call__(self, vars):
        """Evaluate the constraint"""
        return self.eval(vars)

    @abstractmethod
    def eval(self, vars):
        """Concrete implementation of the evaluator called by __call__"""

    @abstractmethod
    def lower_bound(self):
        """Returns the lower bound of the constraint"""

    @abstractmethod
    def upper_bound(self):
        """Returns the upper bound of the constraint"""

    @property
    def slack(self):
        return self.__slack

    @slack.setter
    def slack(self, val):
        if (type(val) is int or type(val) is float) and val >= 0.:
            self.__slack = val
        else:
            raise ValueError("slack must be a nonnegative numeric value")

class CostRelaxedNonlinearComplementarity(ComplementarityFunction):
    """
        For the nonlinear complementarity constraint:
            f(x) >= 0
            z >= 0
            z*f(x) = 0
        This class implements the nonnegative inequality constraints as the constraint:
            f(x) >= 0
            z >= 0
        and provides a separate method call to include the product constraint:
            z*f(x) = 0
        as a cost. The parameter cost_weight sets the penalty parameter for the cost function 
    """
    def __init__(self, fcn=None, xdim=0, zdim=1):
        super().__init__(fcn, xdim, zdim)
        self.cost_weight = 1.
    
    def eval(self, vars):
        """
        Evaluate the inequality constraints only
        
        The argument must be a numpy array of decision variables ordered as [x, z]
        """
        x, z = np.split(vars, [self.xdim])
        return np.concatenate([self.fcn(x), z], axis=0)

    def lower_bound(self):
        """Return the lower bound"""
        return np.zeros((2*self.zdim))

    def upper_bound(self):
        """Return the upper bound"""
        return np.full((2*self.zdim,), np.inf)

    def product_cost(self, vars):
        """
        Returns the product constraint as a scalar for use as a cost

        The argument must be a numpy array of decision variables organized as [x, z]
        """
        x, z = np.split(vars, [self.xdim])
        return self.cost_weight * z.dot(self.fcn(x))

    @property
    def cost_weight(self):
        return self.__cost_weight

    @cost_weight.setter
    def cost_weight(self, val):
        if (type(val) == 'int' or type(val) == 'float') and val >= 0.:
            self.__cost_weight = val
        else:
            raise ValueError("cost_weight must be a nonnegative numeric value")       

class ConstantSlackNonlinearComplementarity(ComplementarityFunction):
    """
    Implements the nonlinear complementarity constraint with a constant slack variable
    Implements the problem as:
        f(x) >= 0
        z >= 0 
        z*f(x) <= s
    In this implementation, the slack is pushed to the upper bound of the constraints
    """
    def eval(self, vars):
        """ 
        Evaluates the original nonlinear complementarity constraint 
        
        The argument must be a numpy array of decision variables organized as [x, z]
        """
        x, z = np.split(vars, [self.xdim])
        fcn_val = self.fcn(x)
        return np.concatenate([fcn_val, z, fcn_val*z], axis=0)

    def upper_bound(self):
        """ Returns the upper bound of the constraint"""
        return np.concatenate([np.full((2*self.zdim, ), np.inf), self.slack * np.ones((self.zdim,))], axis=0)

    def lower_bound(self):
        """ Returns the lower bound of the constraint"""
        return np.concatenate([np.zeros((2*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class ChanceConstrainedComplementarityNONLINEAR(ConstantSlackNonlinearComplementarity):
    """
    Implements the nonlinear complementarity constraint with a constant slack variable
    Implements the problem as:
        f(x) - lb >= 0
        z >= 0 
        z*f(x) - ub <= s
    In this implementation, the slack is pushed to the upper bound of the constraints
    """
    def __init__(self,  fcn, xdim = 0, zdim = 1, beta = 0.5, theta = 0.5, sigma = 0):
        super(ConstantSlackNonlinearComplementarity, self).__init__(fcn, xdim, zdim)
        # self.fcn = fcn
        # self.xdim = xdim
        # self.zdim = zdim
        # self.slack = 0.
        self.beta = beta
        self.theta = theta
        self.sigma = sigma
        self.lb, self.ub = self.chance_constraint()

    def eval(self, vars):
        """ 
        Evaluates the original nonlinear complementarity constraint 
        
        The argument must be a numpy array of decision variables organized as [x, z]
        """
        x, z = np.split(vars, [self.xdim])
        fcn_val = self.fcn(x)
        return np.concatenate([fcn_val - self.lb, z, fcn_val*z-self.ub], axis=0)

    def chance_constraint(self):
        '''
        This method implements chance constraint
        Output:
            [lower_bound, upper_bound]
        '''
        lb = -np.sqrt(2)*self.sigma*erfinv(2* self.beta - 1)
        ub = -np.sqrt(2)*self.sigma*erfinv(1 - 2*self.theta)
        return lb, ub

class VariableSlackNonlinearComplementarity(ComplementarityFunction):
    """
    Implements the nonlinear complementarity constraint as
        f(x) >= 0
        z >= 0 
        z*f(x) - s <= 0
    where s is a decision variable. In this implementation, the bounds on the constraint are fixed
    """
    def eval(self, vars):
        """
        Evaluate the complementarity constraint
        
        The argument must be a numpy array of decision variables [x, z, s]
        """
        x, z, s = np.split(vars, np.cumsum([self.xdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate([fcn_val, z, fcn_val*z -s], axis=0)

    def upper_bound(self):
        """Return the lower bound"""
        return np.concatenate([np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the upper bound"""
        return np.concatenate([np.zeros((2*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class ConstantSlackLinearEqualityComplementarity(ComplementarityFunction):
    """
    Introduces new variables and an equality constraint to implement the nonlinear constraint as a linear complementarity constraint with a nonlinear equality constraint. The original problem is implemented as:
        r - f(x) = 0
        r >= 0, z>= 0
        r*z <= s
    where r is the extra set of variables and s is a constant slack added to the upper bound
    """
    def eval(self, vars):
        """
        Evaluate the constraint
        
        The argument must be a numpy array of decision variables ordered as [x, z, r]
        """
        x, z, r = np.split(vars, np.cumsum([self.xdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate((r-fcn_val, r, z, r*z), axis=0)

    def upper_bound(self):
        """Return the upper bound of the constraint"""
        return np.concatenate([np.zeros((self.zdim,)), np.full((2*self.zdim,), np.inf), self.slack*np.ones((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the lower bound of the constraint"""
        return np.concatenate([np.zeros((3*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class ChanceConstrainedComplementarityLINEAR(ConstantSlackLinearEqualityComplementarity):
    """
    Introduces new variables and an equality constraint to implement the nonlinear constraint 
    as a linear complementarity constraint with a nonlinear equality constraint with chance constraints. 
    The original problem is implemented as:
        r - f(x) = 0
        r - lb >= 0, z>= 0
        r*z - ub <= s
    where r is the extra set of variables and s is a constant slack added to the upper bound
    """
    def __init__(self,  fcn, xdim = 0, zdim = 1, beta = 0.5, theta = 0.5, sigma = 0):
        super(ConstantSlackLinearEqualityComplementarity, self).__init__(fcn, xdim, zdim)
        # self.fcn = fcn
        # self.xdim = xdim
        # self.zdim = zdim
        # self.slack = 0.
        self.beta = beta
        self.theta = theta
        self.sigma = sigma
        self.lb, self.ub = self.chance_constraint()

    def eval(self, vars):
        """
        Evaluate the constraint
        
        The argument must be a numpy array of decision variables ordered as [x, z, r]
        """
        x, z, r = np.split(vars, np.cumsum([self.xdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate((r-fcn_val, r - self.lb, z, r*z - self.ub), axis=0)

    def chance_constraint(self):
        '''
        This method implements chance constraint
        Output:
            [lower_bound, upper_bound]
        '''
        lb = -np.sqrt(2)*self.sigma*erfinv(2* self.beta - 1)
        ub = -np.sqrt(2)*self.sigma*erfinv(1 - 2*self.theta)
        return lb, ub

class VariableSlackLinearEqualityComplementarity(ComplementarityFunction):
    """
    Introduces new variables and an equality constraint to implement the nonlinear constraint as a linear complementarity constraint with a nonlinear equality constraint. The original problem is implemented as:
        r - f(x) = 0
        r >= 0, z>= 0
        r*z -s <= 0
    where r is the extra set of variables and s is a variable slack
    """
    def eval(self, vars):
        """
        Evaluate the constraint. 
        The arguments must be a numpy array of decision variables including:
            [x, z, r, s]
        """
        x, z, r, s = np.split(vars, np.cumsum([self.xdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate((r-fcn_val, r, z, r*z - s), axis=0)

    def upper_bound(self):
        """Return the upper bound of the constraint"""
        return np.concatenate([np.zeros((self.zdim,)), np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the lower bound of the constraint"""
        return np.concatenate([np.zeros((3*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class NonlinearComplementarityFcn():
    """
    Implements a complementarity relationship involving a nonlinear function, such that:
        f(x) >= 0
        z >= 0
        f(x)*z <= s
    where f is the function, x and z are decision variables, and s is a slack parameter.
    By default s = 0 (strict complementarity)
    """
    def __init__(self, fcn, xdim=0, zdim=1, 
                        slacktype=NCCSlackType.CONSTANT_SLACK, 
                        ncc_impl=NCCImplementation.NONLINEAR):
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.slack = 0.
        self._eval = self._get_eval_function(ncc_impl)
        self._split_variables = self._get_variable_splitter(slacktype)
        self.cost_weight = 1.
    
    def __call__(self, vars):
        """Evaluate the complementarity constraint """
        x, z, s = self.split_vars(vars, [self.xdim, self.zdim])
        fcn_val = self.fcn(x)
        return np.concatenate((fcn_val,z, fcn_val * z - self.slack), axis=0)

    def lower_bound(self):
        return np.concatenate((np.zeros((2*self.zdim,)), -np.full((self.zdim,), np.inf)), axis=0)
    
    def upper_bound(self):
        return np.concatenate((np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))), axis=0)

    @property
    def slack(self):
        return self.__slack

    @slack.setter
    def slack(self, val):
        if (type(val) == 'int' or type(val) == 'float') and val >= 0.:
            self.__slack = val
        else:
            raise ValueError("slack must be a nonnegative numeric value")