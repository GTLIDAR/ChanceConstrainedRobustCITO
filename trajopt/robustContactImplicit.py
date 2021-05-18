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
from trajopt.constraints import NonlinearComplementarityFcn
import warnings
import math
from trajopt.constraints import ConstantSlackNonlinearComplementarity, ComplementarityFactory, NCCImplementation, NCCSlackType, ChanceConstrainedComplementarityLINEAR
from trajopt.contactimplicit import OptimizationOptions, DecisionVariableList
class ChanceConstrainedContactImplicit(ContactImplicitDirectTranscription):
    '''
    This class implements chance constrained 
    relaxation for contact implicit trajectory 
    optimization with optional uncertainty settings.
    '''

    def __init__(self, plant, context, 
                        num_time_samples = 101,
                        # duration = 1, 
                        minimum_timestep = 0.01, 
                        maximum_timestep = 0.01, 
                        chance_param = [0.5, 0.5, 0], 
                        distance_param = [0.1, 1], 
                        friction_param = [0.1, 0.01, 1], 
                        optionERM = 1, 
                        optionCC = 1,
                        options = None):
        # add chance constraint variables
        # self.chance_param = chance_param
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
        self.erm_option = optionERM
        # Chance Constraint options:
        #   option 1: strict contact constraints
        #   option 2: chance constraint relaxation for normal distance
        #   option 3: chance constraint relaxation for friction cone
        #   option 4: chance constraint relaxation for normal distance and friction cone
        self.cc_option = optionCC
        super(ChanceConstrainedContactImplicit, self).__init__(plant, context, num_time_samples, minimum_timestep, maximum_timestep, options)
        
        # Uncertainty options:
        #   option 1: no uncertainty
        #   option 2: uncertainty from normal distance
        #   option 3: unvertainty from friction cone
        #   option 4: uncertainty from both normal distance And friction cone
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

    def _add_contact_constraints(self):
        """ Add complementarity constraints for contact to the optimization problem"""
        # self.sliding_cstr = NonlinearComplementarityFcn(self._sliding_velocity,
        #                                             xdim = self.x.shape[0] + self.numN,
        #                                             zdim = self.numT,
        #                                             slack = 0.)
        # self.normalV_cstr = NonlinearComplementarityFcn(self._normal_velocity,
        #                                             xdim = self.x.shape[0] + self.numN + self.numT,
        #                                             zdim = self.numN,
        #                                             slack = 0.)
        numN = self._normal_forces.shape[0]
        numT = self._tangent_forces.shape[0]
        factory = ComplementarityFactory(self.options.ncc_implementation, self.options.slacktype)
        self.sliding_cstr = factory.create(self._sliding_velocity, xdim = self.x.shape[0] + numN, zdim=numT)
        # Determine the variables according to implementation and slacktype options
        self.distance_vars = DecisionVariableList([self.x, self._normal_forces])
        self.sliding_vars = DecisionVariableList([self.x, self._sliding_vel, self._tangent_forces])
        self.friccone_vars = DecisionVariableList([self.x, self._normal_forces, self._tangent_forces, self._sliding_vel])
        # Check and add slack variables
        if self.options.ncc_implementation == NCCImplementation.LINEAR_EQUALITY:
            self.distance_vars.add(self.slacks[0:numN,:])
            self.sliding_vars.add(self.slacks[numN:numN+numT,:])
            self.friccone_vars.add(self.slacks[numN+numT:2*numN+numT,:])
        if self.options.ncc_implementation != NCCImplementation.COST and self.options.slacktype == NCCSlackType.VARIABLE_SLACK:
            self.distance_vars.add(self.slacks[-1,:])
            self.sliding_vars.add(self.slacks[-1,:])
            self.friccone_vars.add(self.slacks[-1,:])

        if self.cc_option is 1:
            self.distance_cstr = factory.create(self._normal_distance, xdim=self.x.shape[0], zdim=numN)
            self.friccone_cstr = factory.create(self._friction_cone, self.x.shape[0] + numN + numT, numN)
            if self.erm_option is 1:
                print("No uncertainty; No chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_friction_cone_constraint(n)
                    self._add_normal_distance_constraint(n)
                    self._add_sliding_velocity_constraint(n) 
            if self.erm_option is 2:
                print("Uncertainty from normal distance; No chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_friction_cone_constraint(n)
                    self._add_sliding_velocity_constraint(n)  
            if self.erm_option is 3:
                print("Uncertainty from friction cone; No chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_normal_distance_constraint(n)
                    self._add_sliding_velocity_constraint(n) 
            if self.erm_option is 4:
                for n in range(0, self.num_time_samples):
                    self._add_sliding_velocity_constraint(n)  

        elif self.cc_option is 2:
            print("Normal distance contact constraint relaxed")
            self.distance_cstr = ChanceConstrainedComplementarityLINEAR(self._normal_distance, xdim=self.x.shape[0], zdim=numN, beta = self.beta, theta = self.theta, sigma = self.sigma)
            self.friccone_cstr = factory.create(self._friction_cone, self.x.shape[0] + numN + numT, numN)
            if self.erm_option is 1:
                print("No uncertainty; Normal distance chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_friction_cone_constraint(n)
                    self._add_normal_distance_constraint(n)
                    self._add_sliding_velocity_constraint(n) 
            if self.erm_option is 2:
                print("Uncertainty from normal distance; Normal distance chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_friction_cone_constraint(n)
                    self._add_sliding_velocity_constraint(n)
                    self._add_normal_distance_constraint(n)
            if self.erm_option is 3:
                print("Uncertainty from friction cone; Normal distance chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_normal_distance_constraint(n)
                    self._add_sliding_velocity_constraint(n)

        elif self.cc_option is 3:
            print("Friction Cone constraint relaxed")
            self.distance_cstr = NonlinearComplementarityFcn(self._normal_distance,
                                                    xdim = self.x.shape[0],
                                                    zdim = self.numN,
                                                    slack = 0.)
            self.friccone_cstr = self._friction_cone_cc
            if self.erm_option is 1:
                print("No uncertainty; Friction cone chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_friction_cone_constraint(n)
                    self._add_normal_distance_constraint(n)
                    self._add_sliding_velocity_constraint(n) 
            if self.erm_option is 2:
                print("Uncertainty from normal distance; Friction cone chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_friction_cone_constraint(n)
                    self._add_sliding_velocity_constraint(n)
            if self.erm_option is 3:
                print("Uncertainty from friction cone; Friction cone chance constraint relaxation")
                for n in range(0, self.num_time_samples):
                    self._add_normal_distance_constraint(n)
                    self._add_sliding_velocity_constraint(n)
                    self._add_friction_cone_constraint(n)
            
        elif self.cc_option is 4:
            print("Both Normal distance and Friction cone contact constaints relaxed")
            # TODO: This case is not implemented
            self.distance_cstr = self._normal_distance_cc
            self.friccone_cstr = self._friction_cone_cc
            
        else :
            print("Undefined chance constraint option")
            quit()

        # Check for the case of cost-relaxed complementarity
        if self.options.ncc_implementation == NCCImplementation.COST:
            for n in range(0, self.num_time_samples-1):
                # Normal distance cost
                self.prog.AddCost(self.distance_cstr.product_cost, vars=distance_vars.get(n), description = "DistanceProductCost")
                # Sliding velocity cost
                self.prog.AddCost(self.sliding_cstr.product_cost, vars=sliding_vars.get(n), description = "VelocityProductCost")
                # Friction cone cost
                self.prog.AddCost(self.friccone_cstr.product_cost, vars=friccone_vars.get(n), description = "FricConeProductCost")
            self.slack_cost = []
        elif self.options.slacktype == NCCSlackType.VARIABLE_SLACK:
            a = np.ones(self.slacks.shape[1],)
            self.slack_cost = self.prog.AddLinearCost(a=a, vars=self.slacks[-1,:], description="SlackCost")
        else:
            self.slack_cost = []
    
    def _add_normal_distance_constraint(self, n):
        self.prog.AddConstraint(self.distance_cstr, 
                        lb=self.distance_cstr.lower_bound(),
                        ub=self.distance_cstr.upper_bound(),
                        vars=self.distance_vars.get(n),
                        description="normal_distance")

    def _add_sliding_velocity_constraint(self, n):
        # self.prog.AddConstraint(self.sliding_cstr,
        #                 lb=self.sliding_cstr.lower_bound(),
        #                 ub=self.sliding_cstr.upper_bound(),
        #                 vars=np.concatenate((self.x[:,n], self.l[self.numN+self.numT:,n], self.l[self.numN:self.numN+self.numT,n]), axis=0),
        #                 description="sliding_velocity")
        self.prog.AddConstraint(self.sliding_cstr,
                        lb=self.sliding_cstr.lower_bound(),
                        ub=self.sliding_cstr.upper_bound(),
                        vars=self.sliding_vars.get(n),
                        description="sliding_velocity")

    def _add_friction_cone_constraint(self, n):
        self.prog.AddConstraint(self.friccone_cstr, 
                        lb=self.friccone_cstr.lower_bound(),
                        ub=self.friccone_cstr.upper_bound(),
                        vars=self.friccone_vars.get(n),
                        description="friction_cone")

    def _friction_cone_cc(self, vars):
        """
        chance constraint relaxation for friction cone

        Arguments:
            The decision variable list:
                z = [state,normal_forces, friction_forces, velocity_slacks]
        """
        plant, context, _ = self._autodiff_or_float(vars)
        ind = np.cumsum([self.x.shape[0], self.numN, self.numT])
        x, fN, fT, gam = np.split(vars, ind)
        plant.multibody.SetPositionsAndVelocities(context, x)
        mu = plant.GetFrictionCoefficients(context)
        mu = np.diag(mu)
        r1 = mu.dot(fN) - self._e.dot(fT)
        sigma = self.sigma * fN
        lb, ub = self._chance_constraint(sigma)
        return np.concatenate((r1- lb, gam, r1*gam - ub*gam))
    
    def _normal_distance_cc(self, vars):
        """
        chance constraint relaxation for normal distance

        Arguments:
            The decision variable list:
                z = [state,normal_forces]
        """
        # Check if the decision variables are floats
        plant, context, _ = self._autodiff_or_float(vars)
        # Split the variables from the decision list
        x, fN = np.split(vars, [self.x.shape[0]])
        # Calculate the normal distance
        plant.multibody.SetPositionsAndVelocities(context, x)    
        phi = plant.GetNormalDistances(context)
        lb, ub = self._chance_constraint(self.sigma)
        return np.concatenate((phi - lb, fN, phi*fN - ub*fN))
    
    def _chance_constraint(self, sigma):
        '''
        Calculates chance constraint lower and upper bounds
        '''
        lb = -np.sqrt(2)*sigma*erfinv(2* self.beta - 1)
        ub = -np.sqrt(2)*sigma*erfinv(1 - 2*self.theta)
        return lb, ub
        # return np.concatenate((lb, ub), axis = 0)

    def distanceERMCost(self, z):
        """
        ERM cost function for normal distance

        Arguments:
            the decision variable list:
                z = [state, forces]
        """
        
        # Check if the decision variables are floats
        plant, context, _ = self._autodiff_or_float(z)
        ind = np.cumsum([self.x.shape[0], self.numN])
        # Split the variables from the decision list
        x, fN, fT = np.split(z, ind)
        plant.multibody.SetPositionsAndVelocities(context, x)    
        phi = plant.GetNormalDistances(context)
        nContact = len(phi)
        
        assert self.distanceVariance >= 0, "Distribution is degenerative"
        
        sigma =  self.distanceVariance
        f = self.ermCost(fN, phi, sigma)
        f = np.sum(f, axis = 0) * self.ermDistanceMultiplier
        # f = f.item()
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
        # r = self._friction_cone(z)
        r = self.friccone_cstr(z)
        frictionConeDefect = r[0]
        sigma = self.frictionVariance * fN + self.frictionBias
        f = self.ermCost(gam, frictionConeDefect, sigma)
        f = np.sum(f, axis = 0) * self.ermFrictionMultiplier
        # f = 
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
        A = self._erf((x - mean)/(sd * np.sqrt(2)))
        cum_dist = 1/2 *(1 + A)
        return cum_dist
        
    def _erf(self, x):
        # save the sign of x
        sign = np.zeros(x.shape)
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
        return sign*y

class ChanceConstrainedComplementarityFcn(NonlinearComplementarityFcn):
    def __init__(self,  fcn, chance_param = [0.5, 0.5, 0], xdim = 0, zdim = 1, slack = 0):
        super(ChanceConstrainedComplementarityFcn, self).__init__(fcn, xdim, zdim, slack)
        self.beta = chance_param[0]
        self.theta = chance_param[1]
        self.sigma = chance_param[2]
        self._chance_constraint()

    def __call__(self, vars):
        x, z = np.split(vars, [self.xdim])
        fcn_val = self.fcn(x)
        prod = self.ub * z 
        
        return np.concatenate((fcn_val - self.lb, z, fcn_val * z - self.slack - prod), axis=0)

    
    def _chance_constraint(self):
        '''
        This method implements chance constraint
        Output:
            [lower_bound, upper_bound]
        '''
        self.lb = -np.sqrt(2)*self.sigma*erfinv(2* self.beta - 1)
        self.ub = -np.sqrt(2)*self.sigma*erfinv(1 - 2*self.theta)
        
        
class FrictionConeComplementarityFcn(NonlinearComplementarityFcn):
    def __init__(self,  fcn, chance_param = [0.5, 0.5, 0], xdim = 0, zdim = 1, slack = 0):
        super(ChanceConstrainedComplementarityFcn, self).__init__(fcn, xdim, zdim, slack)
        self.beta = chance_param[0]
        self.theta = chance_param[1]
        self.sigma = chance_param[2]
        

    def __call__(self, vars):
        x, z = np.split(vars, [self.xdim])
        fcn_val = self.fcn(x)

        prod = self.ub * z 
        
        return np.concatenate((fcn_val - self.lb, z, fcn_val * z - self.slack - prod), axis=0)

    
    def _chance_constraint(self, sigma):
        '''
        This method implements chance constraint
        Output:
            [lower_bound, upper_bound]
        '''
        self.lb = -np.sqrt(2)*sigma*erfinv(2* self.beta - 1)
        self.ub = -np.sqrt(2)*sigma*erfinv(1 - 2*self.theta)