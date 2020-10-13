"""
ContactImplicitDirectTranscription: Implements Contact Implicit Trajectory Optimization using Backward Euler Integration
    Partially integrated with pyDrake
    subclasses pyDrake's MathematicalProgram to formulate and solve nonlinear programs
    uses pyDrake's MultibodyPlant to represent rigid body dynamics
Luke Drnach
October 5, 2020
"""

import numpy as np 
from pydrake.all import MathematicalProgram, MultibodyForces

class ContactImplicitDirectTranscription():
    """
    """
    def __init__(self, plant, context, num_time_samples, minimum_timestep, maximum_timestep):
        """
        """
        # Store parameters
        self.model = plant
        self.context = context
        self.num_time_samples = num_time_samples
        self.minimum_timestep = minimum_timestep
        self.maximum_timestep = maximum_timestep
        # Create the mathematical program
        self.prog = MathematicalProgram()
        # Add decision variables to the program
        self.__add_decision_varibles()
        # Add dynamic constraints (+ contact constraints)
        self.__add_dynamic_constraints()

    def __add_decision_varibles(self):
        #TODO: Implement GetContactJacobian() / Get sizes of reaction forces
        """
            addDecisionVariables
            adds the decision variables for timesteps, states, controls, reaction forces,
            and joint limits to the mathematical program, but does not initialize the 
            values of the decision variables

            addDecisionVariables is called during object construction
        """
        # Add time variables to the program
        self.h = self.prog.NewContinuousVariables(rows=1, cols=self.num_time_samples-1, name='h')
        # Add state variables to the program
        nX = self.model.plant.num_positions() + self.plant.num_velocities()
        self.x = self.prog.NewContinuousVariables(rows=nX, cols=self.num_time_samples, name='x')
        # Add control variables to the program
        nU = self.model.plant.num_actuators()
        self.u = self.prog.NewContinuousVariables(rows=nU, cols=self.num_time_samples, name='u')
        # Add reaction force variables to the program
        Jn, Jt = self.model.plant.GetContactJacobian()
        self.numN = Jn.shape[0]
        self.numT = Jt.shape[0]
        self.l = self.prog.NewContinuousVariables(rows=2*self.numN+self.numT, cols=self.num_time_samples, name='l')
        # store a matrix for organizing the friction forces
        self._e = np.zeros((self.numN, self.numT))
        nD= self.numT / self.numN
        for n in range(0, self.numN):
            self._e[n, n*nD:(n+1)*nD] = 1
        # And joint limit variables to the program
        qhigh = self.model.plant.GetPositionUpperLimits()
        qlow = self.model.plant.GetPositionLowerLimits()
        # Assert that the joint limits be two-sided
        low_inf = np.isinf(qlow)
        high_inf = np.isinf(qhigh)
        assert low_inf == high_inf
        if not all(low_inf):
            nJL = sum(low_inf)
            self.jl = self.NewContinuousVariables(rows=2*nJL, cols=self.num_time_samples, name='jl')
            self._Jl = np.concatenate([np.eye(nJL), -np.eye(nJL)], axis=0)
            self._liminds = not low_inf
        else:
            self.jl = False
        
    def __add_dynamic_constraints(self):
        """

        """
        #TODO: Add control limit constraints
        # At each knot point, add
        #   Limits on the timesteps - DONE
        #   The dynamic consensus constraints - DONE
        #   The contact complementarity constraints
        #   The joint limit constraints (if applicable)
        #   The control effort constraints (if applicable) 
        # At every knot point
        for n in range(0, self.num_time_steps-2):
            # Add in timestep bounding box constraint
            self.prog.AddBoundingBoxConstraint(self.minimum_timestep, self.maximum_timestep, self.h[n])
            # Add dynamics as constraints 
            self.prog.AddConstraint(self.__backward_dynamics, 
                        lb=np.zeros(shape=(self.x.shape[0],1)),
                        ub=np.zeros(shape=(self.x.shape[0], 1)),
                        vars=np.concatenate((self.h[n], self.x[:,n], self.x[:,n+1], self.u[:,n], self.l[:,n+1]), axis=0),
                        description="dynamics")
            # Add complementarity constraints for contact
            #TODO: Fix upper and lower bounds (output size), and indices, for complementarity constraints
            self.prog.AddConstraint(self.__normal_distance_constraint, 
                        lb=np.zeros(shape=(3*self.numN,)),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[0,n+1], self.l[0:self.numN,n+1]), axis=0),
                        description="normal_distance")
            # Sliding velocity constraint 
            self.prog.AddConstraint(self.__sliding_velocity_constraint,
                        lb=np.zeros((3*self.numT,)),
                        ub=np.concatenate([np.full((2*self.numT,), np.inf), np.zeros((self.numT,))], axis=0),
                        vars=np.concatenate((self.x[:,n+1], self.l[self.numN:,n+1]), axis=0),
                        description="sliding_velocity")
            # Friction cone constraint
            self.prog.AddConstraint(self.__friction_cone_constraint, 
                        lb=np.zeros(shape=(3*self.numN,)),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))],axis=0),
                        vars=np.concatenate((self.x[:,n+1], self.l[:,n+1]), axis=0),
                        description="friction_cone")
            # Add joint limit constraints
            self.prog.AddConstraint(self.__joint_limit_constraint,
                        lb=0,
                        ub=0,
                        vars=np.concatenate((self.x[:,n+1], self.jl[:,n+1]), axis=0),
                        description="joint_limits")

    def __backward_dynamics(self, z):  
        """
        backward_dynamics: Backward Euler integration of the dynamics constraints
        Decision variables are passed in through a list in the order:
            z = [h, x1, x2, u, l, jl]
        Returns the dynamics defect, evaluated using Backward Euler Integration. 
        """
        #TODO: Write case for no joint limits
        #TODO: Add in Multibody Forces (reactions, joint limits) to the Inverse Dynamics
        # Split the variables from the decision variables
        ind = np.cumsum([self.h.shape[0], self.x.shape[0], self.x.shape[0], self.u.shape[0], self.l.shape[0], self.jl.shape[0]])
        h, x1, x2, u, l, jl = np.split(z, ind)
        # Split configuration and velocity from state
        q1, v1 = np.split(x1,2)
        q2, v2 = np.split(x2,2)
        # Discretize generalized acceleration
        dv = (v2 - v1)/h
        # Update the context
        self.model.plant.SetPositionsAndVelocities(self.context, q2, v2)
        # Calculate multibody forces
        # Do inverse dynamics
        mbf = MultibodyForces(self.plant)
        mbf.SetZero()       #For now, set multibody forces to zero
        tau = self.model.plant.CalcInverseDynamics(self.context, dv, mbf)
        # Calc the residual
        B = self.model.plant.MakeActuationMatrix()
        # Expand the actuation matrix in the case of 1 actuator, to avoid errors with np.dot
        if self.model.plant.num_actuators() == 1:
            B = np.exand_dims(B, axis=1)
        # Calculate the residual force
        fv = tau - B.dot(u)
        # Calc position residual from velocity
        dq2 = self.model.plant.MapVelocityToQDot(self.context, v2)
        fq = q2 - q1 - h*dq2
        # Return dynamics defects
        return np.concatenate((fq, fv), axis=0)

    # Complementarity Constraint functions for Contact
    def __normal_distance_constraint(self, z):
        """
        normal_distance_constraint: internal method implementing the complementarity constraint on normal distance
        
        The decision variable list is:
            z = [state, normal_forces]
        """
        #TODO: Implement getter for normal distance
        idx = np.cumsum([self.x.shape[0], self.numN])
        x, fN, _ = np.split(z)
        phi = self.model.get_normal_distances(x)
        return np.concatenate((phi, fN, phi*fN), axis=0)

    def __sliding_velocity_constraint(self, z):
        """
        The decision variable list is:
            z = [state, friction_forces, velocity_slacks]
        """
        #TODO: Implement contact jacobian
        ind = np.cumsum([self.x.shape[0], self.numT, self.numN])
        x, fT, gam = np.split(z, ind)
        # Get the velocity, and convert to qdot
        q, v = np.split(x, 2)
        self.model.plant.SetPositionsAndVelocities(self.context, q, v)
        dq = self.model.plant.MapVelocityToQDot(self.context, v)
        # Get the contact Jacobian
        _, Jt = self.model.GetContactJacobian(q)
        #Match sliding slacks to sliding velocities
        r1 = self._e.T.dot(gam) + Jt.T.dot(dq)
        r2 = fT * r1
        return np.concatenate((r1, fT, r2), axis=0)

    def __friction_cone_constraint(self, z):
        """
        The decision variable list is stored as :
            z = [state,normal_forces, friction_forces, velocity_slacks]
        """
        #TODO: Implement getter for friction coefficient
        ind = np.cumsum([self.x.shape[0], self.numN, self.numT])
        x, fN, fT, gam = np.split(z, ind)
        mu = self.model.get_friction_coefficients(x)
        mu = np.diag(mu)
        # Match friction forces to normal forces
        r1 = mu.dot(fN) - self._e.dot(fT)
        return np.concatenate((r1, gam, r1*gam), axis=0)

    # Joint Limit Constraints
    def __joint_limit_constraints(self, z):
        """
        Decision variable list:
            z = [state, joint_limit_forces]
        """
        # Get configuration and joint limit forces
        q = z[0:self.nJL]
        jl = z[self.nJL:]
        # Calculate distance from limits
        return np.concatenate([qdiff, jl, jl*qdiff], axis=0)
     
    def addRunningCost(self):
        pass

    def addFinalCost(self):
        pass

    def initial_state(self):
        """
        initial_state: helper function that returns the initial state vector
        """
        return self.x[:,0]

    def final_state(self):
        """
        final_state: helper function that returns the final state vector
        """
        return self.x[:,-1]

    def total_time(self):
        """
        total_time: helper function that returns the sum of the timesteps
        """
        return sum(self.h)