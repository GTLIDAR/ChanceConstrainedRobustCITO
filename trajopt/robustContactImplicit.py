import numpy as np 
import sys
from scipy.special import erfinv
from scipy.stats import norm
from pydrake.all import MathematicalProgram
from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.tree import MultibodyForces_
from trajopt.contactimplicit import ContactImplicitDirectTranscription
import warnings
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
        print("correct constraints")
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
        # nX = self.plant_ad.multibody.num_positions() + self.plant_ad.multibody.num_velocities()
        # nQ = self.plant_ad.multibody.num_positions()
        # nL = len(_lambda) # ia lambda np array or list?
        # Check if the decision variables are floats
        plant, context, _ = self._autodiff_or_float(z)
        # Split the variables from the decision list
        x, fN = np.split(z, [self.x.shape[0]])
        plant.multibody.SetPositionsAndVelocities(context, x)    
        phi = plant.GetNormalDistances(context)
        nContact = len(phi)
        # plant, context, _ = self.__autodiff_or_float(z)
        # plant.multibody.SetPositionsAndVelocities(context, x)
        # dq = plant.multibody.MapVelocityToQDot(context, v)
        
        # Get the contact Jacobian
        Jn, _ = plant.GetContactJacobians(context)
        # print(self.heightVariance)
        assert self.heightVariance > 0, "Distribution is degenerative"
        # print(nContact)
        # print(np.ones(3))
        # print(np.ones(nContact))
        # print(self.heightVariance)
        sigma = np.ones(nContact) * self.heightVariance
        # print(type(sigma))
        # print(type(phi))
        # print(type(fN))
        f = self.ermCost(fN, phi, sigma)
        return f * self.ermMultiplier

    def ermCost(self, x, mu, sigma):
        """
        Gaussian ERM implementation
        """
        # print(x)
        x = x[:]
        mu = mu[:]
        sigma = sigma[:]
        # nX = len(x)
        # print(x)
        # print(mu)
        # print(sigma)
        # Hf = np.zeros(nX, 3*nX, 3*nX)
        
        # check for degenerate distributions
        # check sigma prior
        
        

        # initialize pdf and cdf
        pdf = norm.pdf(x, mu, sigma)
        cdf = norm.cdf(x, mu, sigma)

        # cdf[degenerate and (x > mu)] = 1

        f = np.square(x) - np.square(sigma) * (x + mu) * pdf + (np.square(sigma) + np.square(mu) - np.square(x)) * cdf
        return f
    
    