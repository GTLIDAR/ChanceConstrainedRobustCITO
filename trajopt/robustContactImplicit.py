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
    '''
    This class implements chance constrained 
    relaxation for contact implicit trajectory 
    optimization with optional uncertainty settings.
    '''

    def __init__(self, plant, context, 
                        num_time_samples = 101, 
                        minimum_timestep = 0.01, 
                        maximum_timestep = 0.01, 
                        chance_param = [0.5, 0.5, 0], 
                        distance_param = [0.1, 1], 
                        friction_param = [0.1, 0.01, 1], 
                        optionERM = 1, 
                        optionCC = 1):
        # add chance constraint variables
        self.beta = chance_param[0]
        self.theta = chance_param[1]
        self.sigma = chance_param[2]
        # add distance ERM variables
        self.distanceVariance = distance_param[0]
        self.ermDistanceMultiplier = distance_param[1]
        # add friction ERM variables
        self.frictionVariance = friction_param[0]
        self.frictionBias = friction_param[1]
        self.ermFrictionMultiplier = friction_param[2]
        # Chance Constraint options:
        #   option 1: strict contact constraints
        #   option 2: chance constraint relaxation for normal distance
        #   option 3: chance constraint relaxation for friction cone
        #   option 4: chance constraint relaxation for normal distance and friction cone

        self.cc_option = optionCC
        super(ChanceConstrainedContactImplicit, self).__init__(plant, context, num_time_samples, minimum_timestep, maximum_timestep)
        # Uncertainty options:
        #   option 1: no uncertainty
        #   option 2: uncertainty from normal distance
        #   option 3: unvertainty from friction cone
        #   option 4: uncertainty from both normal distance And friction cone
        self.erm_option = optionERM
        
        if self.erm_option is 1:
            print("Nominal case")
        elif self.erm_option is 2:
            print("Uncertainty from normal distance")
            distanceErmCost = lambda z: self.distanceERMCost(z)
            self.add_running_cost(distanceErmCost,  [self.x, self.l], name = "DistanceERMCost")
        elif self.erm_option is 3:
            print("Uncertainty from fricion cone")
            frictionConeErmCost = lambda z: self.frictionConeERMCost(z)
            self.add_running_cost(frictionConeErmCost,  [self.x, self.l], name = "FrictionConeERMCost")
        elif self.erm_option is 4:
            print("Uncertainty from both normal distance and fricion cone")
            distanceErmCost = lambda z: self.distanceERMCost(z)
            self.add_running_cost(distanceErmCost,  [self.x, self.l], name = "DistanceERMCost")
            frictionConeErmCost = lambda z: self.frictionConeERMCost(z)
            self.add_running_cost(frictionConeErmCost,  [self.x, self.l], name = "FrictionConeERMCost")
        else:
            print("Undefined chance constraint option")
            quit()

    def _add_decision_variables(self):
        """
            adds the decision variables for timesteps, states, controls, reaction forces,
            and joint limits to the mathematical program, but does not initialize the 
            values of the decision variables. Store decision variable lists

            addDecisionVariables is called during object construction
        """
        super(ChanceConstrainedContactImplicit, self)._add_decision_variables() 
        self.lower_bound, self.upper_bound = self._chance_constraint()
    
    def _add_normal_distance_constraint(self, n):
        self.prog.AddConstraint(self._normal_distance_constraint, 
                        lb=np.concatenate([np.zeros((2*self.numN,)), -np.full((self.numN,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[0:self.numN,n]), axis=0),
                        description="normal_distance")

    def _add_normal_distance_constraint_relaxed(self, n):
        self.prog.AddConstraint(self._normal_distance_constraint, 
                        lb = np.concatenate([np.full((self.numN,), self.lower_bound), np.zeros((self.numN,)), -np.full((self.numN,), np.inf)], axis = 0),
                        ub = np.concatenate([np.full((self.numN,), self.upper_bound), np.full((self.numN,), np.inf), np.zeros((self.numN,))], axis = 0),
                        vars=np.concatenate((self.x[:,n], self.l[0:self.numN,n]), axis=0),
                        description="normal_distance")

    def _add_friction_cone_constraint(self, n):
        self.prog.AddConstraint(self._friction_cone_constraint, 
                        lb=np.concatenate([np.zeros((2*self.numN,)),-np.full((self.numN,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="friction_cone")

    def _add_friction_cone_constraint_relaxed(self, n):
        self.prog.AddConstraint(self._friction_cone_constraint, 
                        lb = np.concatenate([np.full((self.numN,), self.lower_bound), np.zeros((self.numN,)), -np.full((self.numN,), np.inf)], axis = 0),
                        ub = np.concatenate([np.full((self.numN,), self.upper_bound), np.full((self.numN,), np.inf), np.zeros((self.numN,))], axis = 0),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="friction_cone")
    
    def _add_sliding_velocity_constraint(self, n):
        self.prog.AddConstraint(self._sliding_velocity_constraint,
                        lb=np.concatenate([np.zeros((2*self.numT,)), -np.full((self.numT,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numT,), np.inf), np.zeros((self.numT,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[self.numN:,n]), axis=0),
                        description="sliding_velocity")

    def _add_normal_velocity_constraint(self, n):
        self.prog.AddConstraint(self._normal_velocity_constraint, 
                        lb=np.concatenate([np.zeros((2*self.numN,)),-np.full((self.numN,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="normal_velocity")
    
    def _add_contact_constraints(self):
        """ Add complementarity constraints for contact to the optimization problem"""
        if self.cc_option is 1:
            print("Strict contact constraints")
            for n in range(0, self.num_time_samples):
                self._add_normal_distance_constraint(n)
                self._add_sliding_velocity_constraint(n)
                self._add_friction_cone_constraint(n)
                # self._add_normal_velocity_constraint(n)
        elif self.cc_option is 2:
            print("Normal distance contact constraint relaxed")
            for n in range(0, self.num_time_samples):
                self._add_normal_distance_constraint_relaxed(n)
                self._add_sliding_velocity_constraint(n)
                self._add_friction_cone_constraint(n)
                self._add_normal_velocity_constraint(n)
        elif self.cc_option is 3:
            print("Friction cone contact constraint relaxed")
            for n in range(0, self.num_time_samples):
                self._add_normal_distance_constraint(n)
                self._add_sliding_velocity_constraint(n)
                self._add_friction_cone_constraint_relaxed(n)
                self._add_normal_velocity_constraint(n)
        elif self.cc_option is 4:
            print("Both Normal distance and Friction cone contact constaints relaxed")
            for n in range(0, self.num_time_samples):
                self._add_normal_distance_constraint_relaxed(n)
                self._add_sliding_velocity_constraint(n)
                self._add_friction_cone_constraint_relaxed(n)
                self._add_normal_velocity_constraint(n)
        else:
            print("Undefined chance constraint option")
            quit()
    
    def _chance_constraint(self):
        '''
        This method implements chance constraint
        Output:
            [lower_bound, upper_bound]

        '''
        lower_bound = -np.sqrt(2)*self.sigma*erfinv(2* self.beta - 1)
        upper_bound = -np.sqrt(2)*self.sigma*erfinv(1 - 2*self.theta)
        return [lower_bound, upper_bound]

    def distanceERMCost(self, z):
        """
        ERM cost function for normal distance

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
        
        assert self.distanceVariance >= 0, "Distribution is degenerative"
        
        sigma = np.ones(nContact) * self.distanceVariance
        
        f = self.ermCost(fN, phi, sigma)
        f = np.sum(f, axis = 0) * self.ermDistanceMultiplier
        return f

    def frictionConeERMCost(self, z):
        """
        ERM cost function for friction cone

        Arguments:
            The decision variable list:
                z = [state,normal_forces, friction_forces, velocity_slacks]
        """
        plant, context, _ = self._autodiff_or_float(z)
        ind = np.cumsum([self.x.shape[0], self.numN, self.numT])
        x, fN, fT, gam = np.split(z, ind)
        r = self._friction_cone_constraint(z)
        frictionConeDefect = r[0]
        sigma = self.frictionVariance * fN + self.frictionBias
        f = self.ermCost(gam, frictionConeDefect, sigma)
        f = np.sum(f, axis = 0) * self.ermFrictionMultiplier
        return f

    def ermCost(self, x, mu, sigma):
        """
        Gaussian ERM implementation
        """
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
        sign = np.zeros(len(x),)
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

    