"""
contactimplicit: Implements Contact Implicit Trajectory Optimization using Backward Euler Integration
    Partially integrated with pyDrake
    contains pyDrake's MathematicalProgram to formulate and solve nonlinear programs
    uses pyDrake's MultibodyPlant to represent rigid body dynamics
Luke Drnach
October 5, 2020
"""
import numpy as np 
from pydrake.all import MathematicalProgram
from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.tree import MultibodyForces_

class ContactImplicitDirectTranscription():
    """
    Implements contact-implicit trajectory optimization using Direct Transcription
    """
    def __init__(self, plant, context, num_time_samples, minimum_timestep, maximum_timestep):
        """
        Create MathematicalProgram with decision variables and add constraints for the rigid body dynamics, contact conditions, and joint limit constraints

            Arguments:
                plant: a TimeSteppingMultibodyPlant model
                context: a Context for the MultibodyPlant in TimeSteppingMultibodyPlant
                num_time_samples: (int) the number of knot points to use
                minimum_timestep: (float) the minimum timestep between knot points
                maximum_timestep: (float) the maximum timestep between knot points
        """
        # Store parameters
        self.plant_f = plant
        self.context_f = context
        self.num_time_samples = num_time_samples
        self.minimum_timestep = minimum_timestep
        self.maximum_timestep = maximum_timestep
        # Create a copy of the plant and context with scalar type AutoDiffXd
        self.plant_f.multibody.SetDefaultContext(context)
        self.plant_ad = self.plant_f.toAutoDiffXd()       
        self.context_ad = self.plant_ad.multibody.CreateDefaultContext()
        # Create MultibodyForces
        MBF = MultibodyForces_[float]
        self.mbf_f = MBF(self.plant_f.multibody)
        MBF_AD = MultibodyForces_[AutoDiffXd]
        self.mbf_ad = MBF_AD(self.plant_ad.multibody)
        # Create the mathematical program
        self.prog = MathematicalProgram()
        # Add decision variables to the program
        self.__add_decision_varibles()
        # Add dynamic constraints 
        self.__add_dynamic_constraints()
        # Add contact constraints
        self.__add_contact_constraints()
        # Initialize the timesteps
        self.__set_initial_timesteps()

    def __add_decision_varibles(self):
        """
            adds the decision variables for timesteps, states, controls, reaction forces,
            and joint limits to the mathematical program, but does not initialize the 
            values of the decision variables. Store decision variable lists

            addDecisionVariables is called during object construction
        """
        # Add time variables to the program
        self.h = self.prog.NewContinuousVariables(rows=self.num_time_samples-1, cols=1, name='h')
        # Add state variables to the program
        nX = self.plant_ad.multibody.num_positions() + self.plant_ad.multibody.num_velocities()
        self.x = self.prog.NewContinuousVariables(rows=nX, cols=self.num_time_samples, name='x')
        # Add control variables to the program
        nU = self.plant_ad.multibody.num_actuators()
        self.u = self.prog.NewContinuousVariables(rows=nU, cols=self.num_time_samples, name='u')
        # Add reaction force variables to the program
        Jn, Jt = self.plant_ad.GetContactJacobians(self.context_ad)
        self.numN = Jn.shape[0]
        self.numT = Jt.shape[0]
        self.l = self.prog.NewContinuousVariables(rows=2*self.numN+self.numT, cols=self.num_time_samples, name='l')
        # store a matrix for organizing the friction forces
        self._e = np.zeros((self.numN, self.numT))
        nD = int(self.numT / self.numN)
        for n in range(0, self.numN):
            self._e[n, n*nD:(n+1)*nD] = 1
        # And joint limit variables to the program
        qhigh = self.plant_ad.multibody.GetPositionUpperLimits()
        qlow = self.plant_ad.multibody.GetPositionLowerLimits()
        # Assume that the joint limits be two-sided
        low_inf = np.isinf(qlow)
        high_inf = np.isinf(qhigh)
        assert all(low_inf == high_inf), "Joint limits must be two-sided"
        if not all(low_inf):
            nJL = sum(low_inf)
            self.jl = self.NewContinuousVariables(rows=2*nJL, cols=self.num_time_samples, name='jl')
            self._Jl = np.concatenate([np.eye(nJL), -np.eye(nJL)], axis=0)
            self._liminds = not low_inf
        else:
            self.jl = False
        
    def __add_dynamic_constraints(self):
        """Add constraints to enforce rigid body dynamics and joint limits"""
        # At each knot point, add
        #   Bounding box constraints on the timesteps
        #   Equality constraints enforcing the dynamics
        for n in range(0, self.num_time_samples-1):
            # Add in timestep bounding box constraint
            self.prog.AddBoundingBoxConstraint(self.minimum_timestep, self.maximum_timestep, self.h[n,:])
            # Add joint limit constraints
            if self.jl:
                # Add dynamics as constraints 
                self.prog.AddConstraint(self.__backward_dynamics, 
                            lb=np.zeros(shape=(self.x.shape[0], 1)),
                            ub=np.zeros(shape=(self.x.shape[0], 1)),
                            vars=np.concatenate((self.h[n,:], self.x[:,n], self.x[:,n+1], self.u[:,n], self.l[:,n+1], self.jl[:,n+1]), axis=0),
                            description="dynamics")
                # Add joint limit constraints
                self.prog.AddConstraint(self.__joint_limit_constraint,
                        lb=0,
                        ub=0,
                        vars=np.concatenate((self.x[:,n+1], self.jl[:,n+1]), axis=0),
                        description="joint_limits")
            else:
                # Add just dynamics as constraints 
                self.prog.AddConstraint(self.__backward_dynamics, 
                            lb=np.zeros(shape=(self.x.shape[0],1)),
                            ub=np.zeros(shape=(self.x.shape[0], 1)),
                            vars=np.concatenate((self.h[n,:], self.x[:,n], self.x[:,n+1], self.u[:,n], self.l[:,n+1]), axis=0),
                            description="dynamics")        
            
    def __add_contact_constraints(self):
        """ Add complementarity constraints for contact to the optimization problem"""
        # At each knot point, add constraints for normal distance, sliding velocity, and friction cone
        for n in range(0, self.num_time_samples):
            # Add complementarity constraints for contact
            self.prog.AddConstraint(self.__normal_distance_constraint, 
                        lb=np.concatenate([np.zeros((2*self.numN,)), -np.full((self.numN,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[0:self.numN,n]), axis=0),
                        description="normal_distance")
            # Sliding velocity constraint 
            self.prog.AddConstraint(self.__sliding_velocity_constraint,
                        lb=np.concatenate([np.zeros((2*self.numT,)), -np.full((self.numT,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numT,), np.inf), np.zeros((self.numT,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[self.numN:,n]), axis=0),
                        description="sliding_velocity")
            # Friction cone constraint
            self.prog.AddConstraint(self.__friction_cone_constraint, 
                        lb=np.concatenate([np.zeros((2*self.numN,)),-np.full((self.numN,), np.inf)], axis=0),
                        ub=np.concatenate([np.full((2*self.numN,), np.inf), np.zeros((self.numN,))], axis=0),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="friction_cone")

    def __backward_dynamics(self, z):  
        """
        backward_dynamics: Backward Euler integration of the dynamics constraints
        Decision variables are passed in through a list in the order:
            z = [h, x1, x2, u, l, jl]
        Returns the dynamics defect, evaluated using Backward Euler Integration. 
        """
        #NOTE: Cannot use MultibodyForces.mutable_generalized_forces with AutodiffXd. Numpy throws an exception
        plant, context, mbf = self.__autodiff_or_float(z)
        # Split the variables from the decision variables
        ind = np.cumsum([self.h.shape[1], self.x.shape[0], self.x.shape[0], self.u.shape[0]])
        h, x1, x2, u, l = np.split(z, ind)
        # Split configuration and velocity from state
        q1, v1 = np.split(x1,2)
        q2, v2 = np.split(x2,2)
        # Discretize generalized acceleration
        dv = (v2 - v1)/h
        # Update the context
        plant.multibody.SetPositionsAndVelocities(context, x2)
        # Set mutlibodyForces to zero
        mbf.SetZero()
        # calculate generalized forces
        B = plant.multibody.MakeActuationMatrix()
        forces = B.dot(u)
        # Gravity
        forces[:] = forces[:] + plant.multibody.CalcGravityGeneralizedForces(context)
        # Joint limits
        if self.jl:
            l, jl = np.split(l, [self.l.shape[0]])
            forces[:] = forces[:] + self._Jl.dot(jl)
        # Ground reaction forces
        Jn, Jt = plant.GetContactJacobians(context)
        J = np.concatenate((Jn, Jt), axis=0)
        forces[:] = forces[:] + J.transpose().dot(l[0:self.numN + self.numT])
        # Do inverse dynamics
        fv = plant.multibody.CalcInverseDynamics(context, dv, mbf) - forces
        # Calc position residual from velocity
        dq2 = plant.multibody.MapVelocityToQDot(context, v2)
        fq = q2 - q1 - h*dq2
        # Return dynamics defects
        return np.concatenate((fq, fv), axis=0)
    
    # Complementarity Constraint functions for Contact
    def __normal_distance_constraint(self, z):
        """
        Complementarity constraint between the normal distance to the terrain and the normal reaction force
        
        Arguments:
            The decision variable list:
                z = [state, normal_forces]
        """
        # Check if the decision variables are floats
        plant, context, _ = self.__autodiff_or_float(z)
        # Split the variables from the decision list
        x, fN = np.split(z, [self.x.shape[0]])
        # Calculate the normal distance
        plant.multibody.SetPositionsAndVelocities(context, x)    
        phi = plant.GetNormalDistances(context)
        # Return the normal distances, forces, and complementary product
        return np.concatenate((phi, fN, phi*fN), axis=0)

    def __sliding_velocity_constraint(self, z):
        """
        Complementarity constraint between the relative sliding velocity and the tangential reaction forces

        Arguments:
            The decision variable list:
                z = [state, friction_forces, velocity_slacks]
        """
        plant, context, _ = self.__autodiff_or_float(z)
        # Split variables from the decision list
        x, fT, gam = np.split(z, np.cumsum([self.x.shape[0], self.numT]))
        # Get the velocity, and convert to qdot
        _, v = np.split(x, 2)
        plant.multibody.SetPositionsAndVelocities(context, x)
        dq = plant.multibody.MapVelocityToQDot(context, v)
        # Get the contact Jacobian
        _, Jt = plant.GetContactJacobians(context)
        # Match sliding slacks to sliding velocities
        r1 = self._e.transpose().dot(gam) + Jt.dot(dq)
        r2 = fT * r1
        return np.concatenate((r1, fT, r2), axis=0)

    def __friction_cone_constraint(self, z):
        """
        Complementarity constraint between the relative sliding velocity and the friction cone

        Arguments:
            The decision variable list is stored as :
                z = [state,normal_forces, friction_forces, velocity_slacks]
        """
        plant, context, _ = self.__autodiff_or_float(z)
        ind = np.cumsum([self.x.shape[0], self.numN, self.numT])
        x, fN, fT, gam = np.split(z, ind)
        plant.multibody.SetPositionsAndVelocities(context, x)
        mu = plant.GetFrictionCoefficients(context)
        mu = np.diag(mu)
        # Match friction forces to normal forces
        r1 = mu.dot(fN) - self._e.dot(fT)
        return np.concatenate((r1, gam, r1*gam), axis=0)

    # Joint Limit Constraints
    def __joint_limit_constraint(self, z):
        """
        Complementarity constraint between the position variables and the joint limit forces

        Arguments:
            Decision variable list:
                z = [state, joint_limit_forces]
        """
        plant, _ = self.__autodiff_or_float(z)
        # Get configuration and joint limit forces
        x, jl = np.split(z, [self.x.shape[0]])
        q, _ = np.split(x, 2)
        # Calculate distance from limits
        qmax = plant.multibody.GetPositionUpperLimits()
        qmin = plant.multibody.GetPositionLowerLimits()
        q_valid = np.isfinite(qmax)
        qdiff = np.concatenate((q[q_valid] - qmax[q_valid],
                                qmin[q_valid] - q[q_valid]),
                                axis=0)
        return np.concatenate([qdiff, jl, jl*qdiff], axis=0)
     
    def __autodiff_or_float(self, z):
        """Returns the autodiff or float implementation of model and context based on the dtype of the decision variables"""
        if z.dtype == "float":
            return (self.plant_f, self.context_f, self.mbf_f)
        else:
            return (self.plant_ad, self.context_ad, self.mbf_ad)

    def __set_initial_timesteps(self):
        """Set the initial timesteps to their maximum values"""
        self.prog.SetInitialGuess(self.h, self.maximum_timestep*np.ones(self.h.shape))

    def set_initial_guess(self, xtraj=None, utraj=None, ltraj=None, jltraj=None):
        """Set the initial guess for the decision variables"""
        if xtraj is not None:
            self.prog.SetInitialGuess(self.x, xtraj)
        if utraj is not None:
            self.prog.SetInitialGuess(self.u, utraj)
        if ltraj is not None:
            self.prog.SetInitialGuess(self.l, ltraj)
        if jltraj is not None:
            self.prog.SetInitialGuess(self.jl, jltraj)

    def add_running_cost(self, cost_func, vars=None, name="RunningCost"):
        """Add a running cost to the program"""
        integrated_cost = lambda x: np.array(x[0] * cost_func(x[1:]))
        for n in range(0, self.num_time_samples-1):
            new_vars = [var[:,n] for var in vars]
            new_vars.insert(0, self.h[n,:])
            self.prog.AddCost(integrated_cost, np.concatenate(new_vars,axis=0), description=name)

    def add_final_cost(self, cost_func, vars=None, name="FinalCost"):
        """Add a final cost to the program"""
        if vars is not None:
            vars = np.concatenate(vars,axis=0)
            self.prog.AddCost(cost_func, vars, description=name)
        else:
            self.prog.AddCost(cost_func,description=name)
            
    def add_quadratic_running_cost(self, Q, b, vars=None, name="QuadraticCost"):
        """
        Add a quadratic running cost to the program
        
        Arguments:
            Q (numpy.array[n,n]): a square numpy array of cost weights
            b (numpy.array[n,1]): a vector of offset values
            vars (list): a list of program decision variables subject to the cost
            name (str, optional): a description of the cost function
        """
        integrated_cost = lambda z: z[0]*(z[1:]-b).dot(Q.dot(z[1:]-b))
        for n in range(0, self.num_time_samples-1):
            new_vars = [var[:,n] for var in vars]
            new_vars.insert(0, self.h[n,:])
            self.prog.AddCost(integrated_cost, np.concatenate(new_vars,axis=0), description=name)

    def add_equal_time_constraints(self):
        """impose that all timesteps be equal"""
       # Enforce the constraint with a linear constraint matrix of pairwise differences 
        num_h = self.h.shape[0]
        M = np.eye(num_h-1, num_h) - np.eye(num_h-1, num_h, 1)
        b = np.zeros((num_h-1,))
        self.prog.AddLinearEqualityConstraint(Aeq=M, beq=b, vars=self.h).evaluator().set_description('EqualTimeConstraints')
        
    def add_state_constraint(self, knotpoint, value, subset_index=None):
        """
        add a constraint to the state vector at a particular knotpoint
        
        Arguments:  
            knotpoint (int): the index of the knotpoint at which to add the constraint
            value (numpy.array): an array of constraint values
            subset_index: optional list of indices specifying which state variables are subject to constraint
        """
        #TODO Check the inputs
        A = np.eye(value.shape[0])
        if subset_index is None:
            subset_index = range(0, self.x.shape[0])       
        self.prog.AddLinearEqualityConstraint(Aeq=A, beq=value, vars=self.x[subset_index, knotpoint]).evaluator().set_description("StateConstraint")
            
    def add_control_limits(self, umin, umax):
        """
        adds acutation limit constraints to all knot pints
        
        Arguments:
            umin (numpy.array): array of minimum control effort limits
            umax (numpy.array): array of maximum control effort limits

        umin and umax must as many entries as there are actuators in the problem. If the control has no effort limit, use np.inf
        """
        #TODO check the inputs
        u_valid = np.isfinite(umin)
        for n in range(0, self.num_time_samples):
            self.prog.AddBoundingBoxConstraint(umin[u_valid], umax[u_valid], self.u[n, u_valid]).evaluator().set_description("ControlLimits")

    def initial_state(self):
        """returns the initial state vector"""
        return self.x[:,0]

    def final_state(self):
        """returns the final state vector"""
        return self.x[:,-1]

    def total_time(self):
        """returns the sum of the timesteps"""
        return sum(self.h)

    def get_program(self):
        """returns the stored mathematical program object for use with solve"""
        return self.prog

    def reconstruct_state_trajectory(self, soln):
        """Returns the state trajectory from the solution"""
        return soln.GetSolution(self.x)

    def reconstruct_input_trajectory(self, soln):
        """Returns the input trajectory from the solution"""
        return soln.GetSolution(self.u)
    
    def reconstruct_reaction_force_trajectory(self, soln):
        """Returns the reaction force trajectory from the solution"""
        return soln.GetSolution(self.l)
    
    def reconstruct_limit_force_trajectory(self, soln):
        """Returns the joint limit force trajectory from the solution"""
        if self.jl:
            return soln.GetSolution(self.jl)
        else:
            return None

    def reconstruct_all_trajectories(self, soln):
        """Returns state, input, reaction force, and joint limit force trajectories from the solution"""
        state = self.reconstruct_state_trajectory(soln)
        input = self.reconstruct_input_trajectory(soln)
        lforce = self.reconstruct_reaction_force_trajectory(soln)
        jlforce = self.reconstruct_limit_force_trajectory(soln)
        return (state, input, lforce, jlforce)

    def get_solution_times(self, soln):
        """Returns a vector of times for the knotpoints in the solution"""
        h = soln.GetSolution(self.h)
        t = np.concatenate((np.zeros(1,), h), axis=0)
        return np.cumsum(t)

    def result_to_dict(self, soln):
        """ unpack the trajectories from the program result and store in a dictionary"""
        t = self.get_solution_times(soln)
        x, u, f, jl = self.reconstruct_all_trajectories(soln)
        soln_dict = {"time": t,
                    "state": x,
                    "control": u, 
                    "force": f,
                    "jointlimit": jl,
                    "solver": soln.get_solver_id().name(),
                    "success": soln.is_success(),
                    "exit_code": soln.get_solver_details().info,
                    "final_cost": soln.get_optimal_cost()
                    }
        return soln_dict

