import numpy as np 
import sys
from scipy.special import erfinv
# from scipy.special import erf
# from mpmath import *
from scipy.stats import norm
from pydrake.all import MathematicalProgram
from matplotlib import mlab
from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.tree import MultibodyForces_
from trajopt.contactimplicit import ContactImplicitDirectTranscription
import warnings
import math
class ChanceConstrainedContactImplicit(ContactImplicitDirectTranscription):
    def __init__(self, plant, context, num_time_samples, minimum_timestep, maximum_timestep):
        super(ChanceConstrainedContactImplicit, self).__init__(plant, context, num_time_samples, minimum_timestep, maximum_timestep)
        # ermCost = lambda z: trajopt.distanceERMCost(z)
        # self.add_running_cost(cost, vars = [trajopt.x, trajopt.l], name = "ERMCost")

    def _add_decision_variables(self):
        # super(ChanceConstrainedContactImplicit, self).__add_decision_varibles()
        """
            adds the decision variables for timesteps, states, controls, reaction forces,
            and joint limits to the mathematical program, but does not initialize the 
            values of the decision variables. Store decision variable lists

            addDecisionVariables is called during object construction
        """
        super(ChanceConstrainedContactImplicit, self)._add_decision_variables()
        
        self.beta = 0.6
        self.theta = 0.6
        self.sigma = 0.1
        self.heightVariance = 0.1
        self.ermMultiplier = 1
    def _add_contact_constraints(self):
        """ Add complementarity constraints for contact to the optimization problem"""
        # print("correct constraints")
        # At each knot point, add constraints for normal distance, sliding velocity, and friction cone
        lb_phi = -np.sqrt(2)*self.sigma*erfinv(2* self.beta - 1)
        ub_phi = -np.sqrt(2)*self.sigma*erfinv(1 - 2*self.theta)
        # ub_prod = sys.float_info.max * ub_phi
        for n in range(0, self.num_time_samples):
            # Add complementarity constraints for contact
            self.prog.AddConstraint(self._normal_distance_constraint, 
                        # lb=np.concatenate([np.zeros((2*self.numN,)), -np.full((self.numN,), np.inf)], axis=0),
                        # ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        lb = np.concatenate([np.full((self.numN,), lb_phi), np.zeros((self.numN,)), -np.full((self.numN,), np.inf)], axis = 0),
                        ub = np.concatenate([np.full((self.numN,), np.inf), np.full((self.numN,), np.inf), np.zeros((self.numN,))], axis = 0),
                        # ub = np.concatenate([np.full((3*self.numN,), np.inf)], axis = 0),
                        vars=np.concatenate((self.x[:,n], self.l[0:self.numN,n]), axis=0),
                        description="normal_distance")
            # Sliding velocity constraint 
            self.prog.AddConstraint(self._sliding_velocity_constraint,
                        lb=np.concatenate([np.zeros((2*self.numT,)), -np.full((self.numT,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numT,), np.inf), np.zeros((self.numT,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[self.numN:,n]), axis=0),
                        description="sliding_velocity")
            # Friction cone constraint
            self.prog.AddConstraint(self._friction_cone_constraint, 
                        lb=np.concatenate([np.zeros((2*self.numN,)),-np.full((self.numN,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="friction_cone")
            # Normal Velocity constraint
            self.prog.AddConstraint(self._normal_velocity_constraint, 
                        lb=np.concatenate([np.zeros((2*self.numN,)),-np.full((self.numN,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="normal_velocity")
    # Complementarity Constraint functions for Contact
    
    def distanceERMCost(self, z):
        """
        ERM cost function

        Arguments:
            the decision variable list:
                z = [state, normal_forces]
        """
        
        # Check if the decision variables are floats
        plant, context, _ = self._autodiff_or_float(z)
        # Split the variables from the decision list
        x, fN = np.split(z, [self.x.shape[0]])
        plant.multibody.SetPositionsAndVelocities(context, x)    
        phi = plant.GetNormalDistances(context)
        nContact = len(phi)
        
        assert self.heightVariance > 0, "Distribution is degenerative"
        
        sigma = np.ones(nContact) * self.heightVariance
        
        f = self.ermCost(fN, phi, sigma)
        f = np.sum(f, axis = 0) * self.ermMultiplier
        
        return f

    def ermCost(self, x, mu, sigma):
        """
        Gaussian ERM implementation
        """
        # print(x)
        x = x[:]
        mu = mu[:]
        sigma = sigma[:]
        
        
        # initialize pdf and cdf
        pdf = self._pdf(x, mu, sigma)
        cdf = self._cdf(x, mu, sigma)

        f = x**2 - sigma**2 * (x + mu) * pdf + (sigma**2+ mu**2 - x**2) * cdf
        
        return f
    def _pdf (self, x, mean, sd):
        prob_density = (1/(sd * np.sqrt(2 * np.pi)) ) * np.exp(-0.5*((x-mean)**2/sd**2))
        return prob_density

    def _cdf (self, x, mean, sd):
        cum_dist = np.zeros(len(x))
        # for i in range(len(x)):
        #     A = self._erf((x[i] - mean)/(sd * np.sqrt(2)))
        #     cum_dist[i] = 1/2 *(1 + A)
        A = self._erf((x - mean)/(sd * np.sqrt(2)))
        cum_dist = 1/2 *(1 + A)
        return cum_dist
        
    def _erf(self, x):
    # save the sign of x
        sign = np.zeros(len(x))
        sign[x >= 0] = 1
        sign[x < 0] = -1
        x = abs(x)

        # constants
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        # A&S formula 7.1.26
        t = 1.0/(1.0 + p*x)
        y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
        return sign*y # erf(-x) = -erf(x)