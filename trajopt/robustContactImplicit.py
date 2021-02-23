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
from trajopt.contactimplicit import ContactImplicitDirectTranscription, NonlinearComplementarityFcn
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
        self.chance_param = chance_param
        # self.beta = chance_param[0]
        # self.theta = chance_param[1]
        # self.sigma = chance_param[2]
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
    
    def _add_contact_constraints(self):
        """ Add complementarity constraints for contact to the optimization problem"""
        self.sliding_cstr = NonlinearComplementarityFcn(self._sliding_velocity,
                                                    xdim = self.x.shape[0] + self.numN,
                                                    zdim = self.numT,
                                                    slack = 0.)
        self.normalV_cstr = NonlinearComplementarityFcn(self._normal_velocity,
                                                    xdim = self.x.shape[0] + self.numN + self.numT,
                                                    zdim = self.numN,
                                                    slack = 0.)
        if self.cc_option is 1:
            print("Strict contact constraints")
            self.distance_cstr = NonlinearComplementarityFcn(self._normal_distance,
                                                    xdim = self.x.shape[0],
                                                    zdim = self.numN,
                                                    slack = 0.)
            self.friccone_cstr = NonlinearComplementarityFcn(self._friction_cone, 
                                                    xdim = self.x.shape[0] + self.numN + self.numT,
                                                    zdim = self.numN,
                                                    slack = 0.)
        elif self.cc_option is 2:
            print("Normal distance contact constraint relaxed")
            self.distance_cstr = ChanceConstrainedComplementarityFcn(self._normal_distance,
                                                    chance_param=self.chance_param,
                                                    xdim = self.x.shape[0],
                                                    zdim = self.numN,
                                                    slack = 0.)
            self.friccone_cstr = NonlinearComplementarityFcn(self._friction_cone, 
                                                    xdim = self.x.shape[0] + self.numN + self.numT,
                                                    zdim = self.numN,
                                                    slack = 0.)
        elif self.cc_option is 3:
            print("Friction Cone constraint relaxed")
            self.distance_cstr = NonlinearComplementarityFcn(self._normal_distance,
                                                    xdim = self.x.shape[0],
                                                    zdim = self.numN,
                                                    slack = 0.)
            self.friccone_cstr = ChanceConstrainedComplementarityFcn(self._friction_cone,
                                                    chance_param=self.chance_param,
                                                    xdim = self.x.shape[0] + self.numN + self.numT,
                                                    zdim = self.numN,
                                                    slack = 0.)
        elif self.cc_option is 4:
            print("Both Normal distance and Friction cone contact constaints relaxed")
            self.distance_cstr = ChanceConstrainedComplementarityFcn(self._normal_distance,
                                                    chance_param=self.chance_param,
                                                    xdim = self.x.shape[0],
                                                    zdim = self.numN,
                                                    slack = 0.)
            self.friccone_cstr = ChanceConstrainedComplementarityFcn(self._friction_cone,
                                                    chance_param=self.chance_param,
                                                    xdim = self.x.shape[0] + self.numN + self.numT,
                                                    zdim = self.numN,
                                                    slack = 0.)
        else :
            print("Undefined chance constraint option")
            quit()
        
        for n in range(0, self.num_time_samples):
            # Add complementarity constraints for contact
            self.prog.AddConstraint(self.distance_cstr, 
                        lb=self.distance_cstr.lower_bound(),
                        ub=self.distance_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n], self.l[0:self.numN,n]), axis=0),
                        description="normal_distance")
            # Sliding velocity constraint 
            self.prog.AddConstraint(self.sliding_cstr,
                        lb=self.sliding_cstr.lower_bound(),
                        ub=self.sliding_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n], self.l[self.numN+self.numT:,n], self.l[self.numN:self.numN+self.numT,n]), axis=0),
                        description="sliding_velocity")
            # Friction cone constraint
            self.prog.AddConstraint(self.friccone_cstr, 
                        lb=self.friccone_cstr.lower_bound(),
                        ub=self.friccone_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="friction_cone")
            # Normal Velocity constraint
            self.prog.AddConstraint(self.normalV_cstr, 
                        lb=self.normalV_cstr.lower_bound(),
                        ub=self.normalV_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="normal_velocity")
    
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
        # r = self._friction_cone(z)
        r = self.friccone_cstr(z)
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
        # print(prod)
        return np.concatenate((fcn_val - self.lb, z, fcn_val * z - self.slack - prod), axis=0)

    
    def _chance_constraint(self):
        '''
        This method implements chance constraint
        Output:
            [lower_bound, upper_bound]
        '''
        self.lb = -np.sqrt(2)*self.sigma*erfinv(2* self.beta - 1)
        self.ub = -np.sqrt(2)*self.sigma*erfinv(1 - 2*self.theta)
        # print(self.lb)
        
