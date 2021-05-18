"""
TimeSteppingMultibodyPlant: an container for pyDrake's MultibodyPlant for use with ContactImplicitDirectTranscription

TimeSteppingMultibodyPlant instantiates pyDrake's MultibodyPlant class and adds a few helper methods for use with ContactImplicitDirectTranscription. In building the model, TimeSteppingMultibodyPlant also constructs a SceneGraph and stores the relevant information about the contact geometry stored in the model. TimeSteppingMultibodyPlant also supports arbitrary terrain geometries by adding a custom terrain object to the class. The terrain specification is used to calculate normal distances and contact Jacobians in TimeSteppingMultibodyPlant

This class is not fully integrated into pyDrake. For example, this class uses a custom implementation of the terrain instead of using a terrain model from pyDrake, and likewise uses custom methods to calculate collision distances. In future, the class may be removed and the operations replaced by pure pyDrake operations.

Note that, due to issues with pybind, TimeSteppingMultibodyPlant does NOT subclass MultibodyPlant. Instead, TimeSteppingMultibodyPlant instantiates MultibodyPlant as property called plant.

Luke Drnach
October 9, 2020
"""
import numpy as np
from math import pi
from pydrake.all import MultibodyPlant, DiagramBuilder, SceneGraph,AddMultibodyPlantSceneGraph, JacobianWrtVariable, AngleAxis, RotationMatrix, RigidTransform, MathematicalProgram, Solve
from pydrake.geometry import Role, Sphere
from pydrake.multibody.parsing import Parser
from systems.terrain import FlatTerrain
from utilities import FindResource, printProgramReport
#TODO: Implemet toAutoDiffXd method to convert to autodiff class

class TimeSteppingMultibodyPlant():
    """
    """
    def __init__(self, file=None, terrain=FlatTerrain(), dlevel=1):
        """
        Initialize TimeSteppingMultibodyPlant with a model from a file and an arbitrary terrain geometry. Initialization also welds the first frame in the MultibodyPlant to the world frame
        """
        self.builder = DiagramBuilder()
        self.multibody, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, 0.001)
        # Store the terrain
        self.terrain = terrain
        self._dlevel = dlevel
        # Build the MultibodyPlant from the file, if one exists
        self.model_index = []
        if file is not None:
            # Parse the file
            self.model_index = Parser(self.multibody).AddModelFromFile(FindResource(file))
        # Initialize the collision data
        self.collision_frames = []
        self.collision_poses = []
        self.collision_radius = []

    def Finalize(self):
        """
        Cements the topology of the MultibodyPlant and identifies all available collision geometries. 
        """
        # Finalize the underlying plant model
        self.multibody.Finalize()
        # Idenify and store collision geometries
        self.__store_collision_geometries()

    def num_contacts(self):
        """ returns the number of contact points"""
        return len(self.collision_frames)

    def num_friction(self):
        """ returns the number of friction components"""
        return 4*self.dlevel*self.num_contacts()

    def GetNormalDistances(self, context):   
        """
        Returns an array of signed distances between the contact geometries and the terrain, given the current system context

        Arguments:
            context: a pyDrake MultibodyPlant context
        Return values:
            distances: a numpy array of signed distance values
        """
        qtype = self.multibody.GetPositions(context).dtype
        nCollisions = len(self.collision_frames)
        distances = np.zeros((nCollisions,), dtype=qtype)
        for n in range(0, nCollisions):
            # Transform collision frames to world coordinates
            collision_pt = self.multibody.CalcPointsPositions(context, 
                                        self.collision_frames[n],
                                        self.collision_poses[n].translation(),
                                        self.multibody.world_frame()) 
            # Squeeze collision point (necessary for AutoDiff plants)
            collision_pt = np.squeeze(collision_pt)
            # Calc nearest point on terrain in world coordinates
            terrain_pt = self.terrain.nearest_point(collision_pt)
            # Calc normal distance to terrain   
            terrain_frame = self.terrain.local_frame(terrain_pt)  
            normal = terrain_frame[0,:]
            distances[n] = normal.dot(collision_pt - terrain_pt) - self.collision_radius[n]
        # Return the distances as a single array
        return distances

    def GetContactJacobians(self, context):
        """
        Returns a tuple of numpy arrays representing the normal and tangential contact Jacobians evaluated at each contact point

        Arguments:
            context: a pyDrake MultibodyPlant context
        Return Values
            (Jn, Jt): the tuple of contact Jacobians. Jn represents the normal components and Jt the tangential components
        """
        qtype = self.multibody.GetPositions(context).dtype
        numN = self.num_contacts()
        numT = int(self.num_friction()/numN)
        D = self.friction_discretization_matrix().transpose()
        Jn = np.zeros((numN, self.multibody.num_velocities()), dtype=qtype)
        Jt = np.zeros((numN * numT, self.multibody.num_velocities()), dtype=qtype)
        for n in range(0, numN):
            # Transform collision frames to world coordinates
            collision_pt = self.multibody.CalcPointsPositions(context,
                                        self.collision_frames[n],
                                        self.collision_poses[n].translation(),
                                        self.multibody.world_frame())
            # Calc nearest point on terrain in world coordinates
            terrain_pt = self.terrain.nearest_point(collision_pt)
            # Calc normal distance to terrain   
            terrain_frame = self.terrain.local_frame(terrain_pt)
            normal, tangent = np.split(terrain_frame, [1], axis=0)
            # Discretize to the chosen level 
            Dtangent = D.dot(tangent)
            # Get the contact point Jacobian
            J = self.multibody.CalcJacobianTranslationalVelocity(context,
                 JacobianWrtVariable.kV,
                 self.collision_frames[n],
                 self.collision_poses[n].translation(),
                 self.multibody.world_frame(),
                 self.multibody.world_frame())
            # Calc contact Jacobians
            Jn[n,:] = normal.dot(J)
            Jt[n*numT: (n+1)*numT, :] = Dtangent.dot(J)
        # Return the Jacobians as a tuple of np arrays
        return (Jn, Jt)    

    def GetFrictionCoefficients(self, context):
        """
        Return friction coefficients for nearest point on terrain
        
        Arguments:
            context: the current MultibodyPlant context
        Return Values:
            friction_coeff: list of friction coefficients
        """
        friction_coeff = []
        for frame, pose in zip(self.collision_frames, self.collision_poses):
            # Transform collision frames to world coordinates
            collision_pt = self.multibody.CalcPointsPositions(context, frame, pose.translation(), self.multibody.world_frame())
            # Calc nearest point on terrain in world coordiantes
            terrain_pt = self.terrain.nearest_point(collision_pt)
            friction_coeff.append(self.terrain.get_friction(terrain_pt))
        # Return list of friction coefficients
        return friction_coeff

    def getTerrainPointsAndFrames(self, context):
        """
        Return the nearest points on the terrain and the local coordinate frame

        Arguments:
            context: current MultibodyPlant context
        Return Values:
            terrain_pts: a 3xN array of points on the terrain
            terrain_frames: a 3x3xN array, specifying the local frame of the terrain
        """
        terrain_pts = []
        terrain_frames = []
        for frame, pose in zip(self.collision_frames, self.collision_poses):
            # Calc collision point
            collision_pt = self.multibody.CalcPointsPositions(context, frame, pose.translation(), self.multibody.world_frame())
            # Calc nearest point on terrain in world coordinates
            terrain_pt = self.terrain.nearest_point(collision_pt)
            terrain_pts.append(terrain_pt)
            # Calc local coordinate frame
            terrain_frames.append(self.terrain.local_frame(terrain_pt))

        return (terrain_pts, terrain_frames)

    def toAutoDiffXd(self):
        """Covert the MultibodyPlant to use AutoDiffXd instead of Float"""

        # Create a new TimeSteppingMultibodyPlant model
        copy_ad = TimeSteppingMultibodyPlant(file=None, terrain=self.terrain, dlevel=self._dlevel)
        # Instantiate the plant as the Autodiff version
        copy_ad.multibody = self.multibody.ToAutoDiffXd()
        copy_ad.scene_graph = self.scene_graph.ToAutoDiffXd()
        copy_ad.model_index = self.model_index
        # Store the collision frames to finalize the model
        copy_ad.__store_collision_geometries()
        return copy_ad

    def set_discretization_level(self, dlevel=0):
        """Set the friction discretization level. The default is 0"""
        self._dlevel = dlevel

    def __store_collision_geometries(self):
        """Identifies the collision geometries in the model and stores their parent frame and pose in parent frame in lists"""
        # Create a diagram and a scene graph
        inspector = self.scene_graph.model_inspector()
        # Locate collision geometries and contact points
        body_inds = self.multibody.GetBodyIndices(self.model_index)
        # Get the collision frames for each body in the model
        for body_ind in body_inds:
            body = self.multibody.get_body(body_ind)
            collision_ids = self.multibody.GetCollisionGeometriesForBody(body)
            for id in collision_ids:
                # get and store the collision geometry frames
                frame_name = inspector.GetName(inspector.GetFrameId(id)).split("::")
                self.collision_frames.append(self.multibody.GetFrameByName(frame_name[-1]))
                self.collision_poses.append(inspector.GetPoseInFrame(id))
                # Check for a spherical geometry
                geoms = inspector.GetGeometries(inspector.GetFrameId(id), Role.kProximity)
                shape = inspector.GetShape(geoms[0])
                if type(shape) is Sphere:
                    self.collision_radius.append(shape.radius())
                else:
                    self.collision_radius.append(0.)

    def __discretize_friction(self, normal, tangent):
        """
        Rotates the terrain tangent vectors to discretize the friction cone
        
        Arguments:
            normal:  The terrain normal direction, (1x3) numpy array
            tangent:  The terrain tangent directions, (2x3) numpy array
        Return Values:
            all_tangents: The discretized friction vectors, (2nx3) numpy array

        This method is now deprecated
        """
        # Reflect the current friction basis
        tangent = np.concatenate((tangent, -tangent), axis=0)
        all_tangents = np.zeros((4*(self._dlevel), tangent.shape[1]))
        all_tangents[0:4, :] = tangent
        # Rotate the tangent basis around the normal vector
        for n in range(1, self._dlevel):
            # Create an angle-axis representation of rotation
            R = RotationMatrix(theta_lambda=AngleAxis(angle=n*pi/(2*(self._dlevel)), axis=normal))
            # Apply the rotation matrix
            all_tangents[n*4 : (n+1)*4, :] = R.multiply(tangent.transpose()).transpose()
        return all_tangents

    def simulate(self, h, x0, u=None, N=1):
        
        # Initialize arrays
        nx = x0.shape[0]
        x = np.zeros(shape=(nx, N))
        x[:,0] = x0
        t = np.zeros(shape=(N,))
        nf = 1
        if u is None:
            B = self.multibody.MakeAcutatorMatrix()
            u = np.zeros(shape=(B.shape[1], N))
        context = self.multibody.CreateDefaultContext()
        Jn, Jt = self.GetContactJacobians(context)
        f = np.zeros(shape=(Jn.shape[0] + Jt.shape[0], N))
        # Integration loop
        for n in range(0,N-1):
            f[:,n] = self.contact_impulse(h, x[:,n], u[:,n])
            x[:,n+1] = self.integrate(h, x[:,n], u[:,n], f[:,n])
            t[n + 1] = t[n] + h
            f[:,n] = f[:,n]/h
        return (t, x, f)

    def integrate(self, h, x, u, f):
        # Get the configuration and the velocity
        q, dq = np.split(x,2)
        # Estimate the next configuration, assuming constant velocity
        qhat = q + h * dq
        # Set the context
        context = self.multibody.CreateDefaultContext()
        self.multibody.SetPositions(context, qhat)
        v = self.multibody.MapQDotToVelocity(context, dq)
        self.multibody.SetVelocities(context, v)
        # Get the current system properties
        M = self.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = self.multibody.CalcBiasTerm(context)
        G = self.multibody.CalcGravityGeneralizedForces(context)
        B = self.multibody.MakeActuationMatrix()
        Jn, Jt = self.GetContactJacobians(context) 
        J = np.vstack((Jn, Jt))
        # Calculate the next state
        b = h * (B.dot(u) - C.dot(dq) + G) + J.transpose().dot(f)
        v = np.linalg.solve(M,b)
        dq += v
        q += h * dq
        # Collect the configuration and velocity into a state vector
        return np.concatenate((q,dq), axis=0)

    def contact_impulse(self, h, x, u):
        # Get the configuration and generalized velocity
        q, dq = np.split(x,2)
        # Estimate the configuration at the next time step
        qhat = q + h*dq
        # Get the system parameters
        context = self.multibody.CreateDefaultContext()
        self.multibody.SetPositions(context, q)
        v = self.multibody.MapQDotToVelocity(context, dq) 
        self.multibody.SetVelocities(context, v)
        M = self.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = self.multibody.CalcBiasTerm(context)
        G = self.multibody.CalcGravityGeneralizedForces(context)
        B = self.multibody.MakeActuationMatrix()
        tau = B.dot(u) - C + G
        Jn, Jt = self.GetContactJacobians(context) 
        phi = self.GetNormalDistances(context)
        alpha = Jn.dot(qhat) - phi
        # Calculate the force size from the contact Jacobian
        numT = Jt.shape[0]
        numN = Jn.shape[0]
        S = numT + 2*numN
        # Initialize LCP parameters
        P = np.zeros(shape=(S,S), dtype=float)
        w = np.zeros(shape=(numN, S))
        numF = int(numT/numN)
        for n in range(0, numN):
            w[n, n*numF + numN:numN + (n+1)*numF] = 1
        # Construct LCP matrix
        J = np.vstack((Jn, Jt))
        JM = J.dot(np.linalg.inv(M))
        P[0:numN + numT, 0:numN + numT] = JM.dot(J.transpose())
        P[:, numN + numT:] = w.transpose()
        P[numN + numT:, :] = -w
        P[numN + numT:, 0:numN] = np.diag(self.GetFrictionCoefficients(context))
        #Construct LCP bias vector
        z = np.zeros(shape=(S,), dtype=float)
        z[0:numN+numT] = h * JM.dot(tau) + J.dot(dq)
        #z[0:numN] += (Jn.dot(q) - alpha)/h   
        z[0:numN] += phi/h
        # Solve the LCP for the reaction impluses
        f, status = solve_lcp(P, z)
        if f is None:
            return np.zeros(shape=(numN+numT,))
        else:
            # Strip the slack variables from the LCP solution
            return f[0:numN + numT]

    def get_multibody(self):
        return self.multibody

    def joint_limit_jacobian(self):
        """ 
        Returns a matrix for mapping the positive-only joint limit torques to 


        """
        # Create the Jacobian as if all joints were limited
        nV = self.multibody.num_velocities()
        I = np.eye(nV)
        # Check for infinite limits
        qhigh= self.multibody.GetPositionUpperLimits()
        qlow = self.multibody.GetPositionLowerLimits()
        # Assert two sided limits
        low_inf = np.isinf(qlow)
        assert all(low_inf == np.isinf(qhigh))
        # Make the joint limit Jacobian
        if not all(low_inf):
            # Remove floating base limits
            floating = self.multibody.GetFloatingBaseBodies()
            floating_pos = []
            while len(floating) > 0:
                body = self.multibody.get_body(floating.pop())
                if body.has_quaternion_dofs():
                    floating_pos.append(body.floating_positions_start())
            # Remove entries corresponding to the extra dof from quaternions
            low_inf = np.delete(low_inf, floating_pos)
            low_inf = np.squeeze(low_inf)
            # Zero-out the entries that don't correspond to joint limits
            I[low_inf, low_inf] = 0
            # Remove the columns that don't correspond to joint limits
            I = np.delete(I, low_inf, axis=1)
            return np.concatenate([I, -I], axis=1)
        else:
            return None

    def get_contact_points(self, context):
        """
            Returns a list of positions of contact points expressed in world coordinates given the system context.
        """
        contact_pts = []
        for frame, pose in zip(self.collision_frames, self.collision_poses):
            contact_pts.append(self.multibody.CalcPointsPositions(context, frame, pose.translation(), self.multibody.world_frame())) 
        return contact_pts
    
    def resolve_contact_forces_in_world(self, context, forces):
        """
            Transform non-negative discretized force components used in complementarity constraints into 3-vectors in world coordinates

            Returns a list of (3,) numpy arrays
        """
        # First remove the discretization
        forces = self.resolve_forces(forces)
        # Reorganize forces from (Normal, Tangential) to a list
        force_list = []
        for n in range(0, self.num_contacts()):
            force_list.append(forces[[n, self.num_contacts() + 2*n, self.num_contacts()+2*n + 1]])
        # Transform the forces into world coordinates using the terrain frames
        _, frames = self.getTerrainPointsAndFrames(context)
        world_forces = []
        for force, frame in zip(force_list, frames):
            world_forces.append(frame.dot(force))
        # Return a list of forces in world coordinates
        return world_forces

    def resolve_limit_forces(self, jl_forces):
        """ 
        Combine positive and negative components of joint limit torques
        
        Arguments:
            jl_forces: (2n, m) numpy array

        Return values:
            (n, m) numpy array
        """
        JL = self.joint_limit_jacobian()
        has_limits = np.sum(abs(JL), axis=1)>0
        f_jl = JL.dot(jl_forces)
        if f_jl.ndim > 1:
            return f_jl[has_limits,:]
        else:
            return f_jl[has_limits]

    def duplicator_matrix(self):
        """Returns a matrix of 1s and 0s for combining friction forces or duplicating sliding velocities"""
        numN = self.num_contacts()
        numT = self.num_friction()
        w = np.zeros((numN, numT))
        nD = int(numT / numN)
        for n in range(numN):
            w[n, n*nD:(n+1)*nD] = 1
        return w

    def friction_discretization_matrix(self):
        """ Make a matrix for converting discretized friction into a single vector"""
        n = 4 * self.dlevel
        theta = np.linspace(0,n-1,n) * 2 * np.pi / n
        D = np.vstack((np.cos(theta), np.sin(theta)))
        # Threshold out small values
        D[np.abs(D)<1e-10] = 0
        return D

    def resolve_forces(self, forces):
        """ Convert discretized friction & normal forces into a non-discretized 3-vector"""
        numN = self.num_contacts()
        n = 4*self.dlevel
        fN = forces[0:numN, :]
        fT = forces[numN:numN*(n+1), :]
        D_ = self.friction_discretization_matrix()
        D = np.zeros((2*numN, n*numN))
        for k in range(numN):
            D[2*k:2*k+2, k*n:(k+1)*n] = D_
        ff = D.dot(fT)
        return np.concatenate((fN, ff), axis=0)

    def static_controller(self, qref, verbose=False):
        """ 
        Generates a controller to maintain a static pose
        
        Arguments:
            qref: (N,) numpy array, the static pose to be maintained

        Return Values:
            u: (M,) numpy array, actuations to best achieve static pose
            f: (C,) numpy array, associated normal reaction forces

        static_controller generates the actuations and reaction forces assuming the velocity and accelerations are zero. Thus, the equation to solve is:
            N(q) = B*u + J*f
        where N is a vector of gravitational and conservative generalized forces, B is the actuation selection matrix, and J is the contact-Jacobian transpose.

        Currently, static_controller only considers the effects of the normal forces. Frictional forces are not yet supported
        """
        #TODO: Solve for friction forces as well

        # Check inputs
        if qref.ndim > 1 or qref.shape[0] != self.multibody.num_positions():
            raise ValueError(f"Reference position mut be ({self.multibody.num_positions(),},) array")
        # Set the context
        context = self.multibody.CreateDefaultContext()
        self.multibody.SetPositions(context, qref)
        # Get the necessary properties
        G = self.multibody.CalcGravityGeneralizedForces(context)
        Jn, _ = self.GetContactJacobians(context)
        phi = self.GetNormalDistances(context)
        B = self.multibody.MakeActuationMatrix()
        #Collect terms
        A = np.concatenate([B, Jn.transpose()], axis=1)
        # Create the mathematical program
        prog = MathematicalProgram()
        l_var = prog.NewContinuousVariables(self.num_contacts(), name="forces")
        u_var = prog.NewContinuousVariables(self.multibody.num_actuators(), name="controls")
        # Ensure dynamics approximately satisfied
        prog.AddL2NormCost(A = A, b = -G, vars=np.concatenate([u_var, l_var], axis=0))
        # Enforce normal complementarity
        prog.AddBoundingBoxConstraint(np.zeros(l_var.shape), np.full(l_var.shape, np.inf), l_var)
        prog.AddConstraint(phi.dot(l_var) == 0)
        # Solve
        result = Solve(prog)
        # Check for a solution
        if result.is_success():
            u = result.GetSolution(u_var)
            f = result.GetSolution(l_var)
            if verbose:
                printProgramReport(result, prog)
            return (u, f)
        else:
            print(f"Optimization failed. Returning zeros")
            if verbose:
                printProgramReport(result,prog)
            return (np.zeros(u_var.shape), np.zeros(l_var.shape))

    @property
    def dlevel(self):
        return self._dlevel

    @dlevel.setter
    def dlevel(self, val):
        """Check that the value of dlevel is a positive integer"""
        if type(val) is int and val > 0:
            self._dlevel = val
        else:
            raise ValueError("dlevel must be a positive integer")

def solve_lcp(P, q):
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(q.size)
    prog.AddLinearComplementarityConstraint(P,q,x)
    result = Solve(prog)

    status = result.is_success()
    z = result.GetSolution(x)
    return (z, status)
    
