############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# MultiGridClass
# one dimensional for now.
####
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse

from _2d_getMatrices import get_T, get_R, get_Tint, get_Rint

def iterative_solver(u, rhs, dx, num_iterations=1, method='GS'):
        ''' Approximiating u system Au = rhs using an iterative method
        *input*
        u:  size total boundary, including endpoints, input is initial guess
        rhs: size total boundary, including endpoints, input should be unchanged

        *output*
        u: size unchanges, but closer approximation to the solution'''
        # Set omega for SOR
        omega = 2.0 - 2 * np.pi * dx

        m  = len(rhs)
        for j in range(num_iterations):
            for i in range(1, m - 1):

                # interior jacobi iteration
                if method=='GS':
                    u[i] = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])

                elif method=='SOR':
                    u_gs = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])    
                    u[i] = u[i] + omega * (u_gs - u[i])
        return u

def iterative_solver_2d(u, rhs, dx, num_iterations=1, method='GS'):
        ''' Approximiating u system Au = rhs using an iterative method
        *input*
        u:  size total boundary, including endpoints, input is initial guess
        rhs: size total boundary, including endpoints, input should be unchanged

        *output*
        u: size unchanges, but closer approximation to the solution'''
        # Set omega for SOR
        omega2 = 2.0 - 2 * np.pi * dx

        m  = len(rhs)
        for k in range(num_iterations):
            for i in range(1, m - 1):
                for j in range(1, m - 1):

                    if method=='GS':
                        u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j-1] + u[i, j+1] - dx**2 * rhs[i, j])

                    elif method=='SOR':
                        u_gs = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j-1] + u[i, j+1] - dx**2 * rhs[i, j])
                        u[i, j] = u[i, j] + omega2 * (u_gs - u[i, j])
        return u

def v_sched(u, A, rhs, dx, num_pre=2, num_post=2, level=2, method='GS'):
    ''' Approximiating u system Au = rhs using multigrid techniques
    *input*
    u:  size total boundary, including endpoints, input is initial guess
    rhs: size total boundary, including endpoints, input should be unchanged
    A: dimenssions of the interior of the problem

    *output*
    u: size unchanged, but closer approximation to the solution'''

    ## BASE CASE
    if level == 0:
        # print 'base case'
        u[1:-1] = np.linalg.solve(A, rhs[1:-1])
        return u
    
    # -----------------
    # Relax on Au = rhs (num_pre)
    # -----------------
    u = iterative_solver(u, rhs, dx, num_iterations=num_pre, method='GS')
    
    # -----------------
    # Solve defect eq for residue
    # -----------------
    residue = np.zeros(len(rhs))
    residue[1:-1] = rhs[1:-1] - np.dot(A, u[1:-1])
    R = get_R(len(residue) / 2 + 1 )
    residue = np.dot(R, residue)

    R_int = get_Rint(len(A[::2]) - 1)
    T_int = get_Tint(len(A[::2]) - 1)
    A = np.dot(R_int, np.dot(A, T_int)) #RAT 

    # -----------------
    # Get error and use to improve u
    # -----------------
    e = v_sched(np.zeros(len(residue)), A, residue, 2.0 * dx, num_pre=num_pre, num_post=num_post, level=level-1, method='GS')
    
    T = get_T(len(e))
    e = np.dot(T, e)

    u = u + e

    # -----------------
    # Relax on Au = rhs (num_post)
    # -----------------
    u = iterative_solver(u, rhs, dx, num_iterations=num_post, method='GS')

    return u

def v_sched_2d(u, A, rhs, dx, num_pre=2, num_post=2, level=2, method='GS'):
    ''' Approximiating u system Au = rhs using multigrid techniques
    *input*
    u:  size total boundary, shape is square, including endpoints, input is initial guess
    rhs: size total boundary, shape is square, including endpoints, input should be unchanged
    A: dimenssions of the interior of the problem

    *output*
    u: size unchanged, but closer approximation to the solution'''
    # print A[0,:]
    ## BASE CASE
    if level == 1:
        # print 'base case'
        rhs_vec = rhs[1:-1, 1:-1].reshape((len(rhs)-2)**2)
        u_vec = np.linalg.solve(A, rhs_vec)
        u[1:-1, 1:-1] = u_vec.reshape(len(rhs)-2, len(rhs)-2)
        return u
    
    # -----------------
    # Relax on Au = rhs (num_pre)
    # -----------------
    u = iterative_solver_2d(u, rhs, dx, num_iterations=num_pre, method=method)
    
    # -----------------
    # Solve defect eq for residue
    # -----------------
    residue = np.zeros((len(rhs), len(rhs)))

    rhs_vec = rhs[1:-1, 1:-1].reshape((len(rhs)-2)**2)
    u_vec = u[1:-1, 1:-1].reshape((len(u)-2)**2)
    residue_vec = rhs_vec - np.dot(A, u_vec)
    
    residue[1:-1, 1:-1] = residue_vec.reshape((len(residue)-2, len(residue)-2))

    R = get_R(len(residue)/2 +1, 2)
    # print [sum(R[i, :] for i in range(len(R[:,0])))]
    residue_vec = np.dot(R, residue.reshape(len(residue)**2))
    residue = residue_vec.reshape((len(residue)/2+1, len(residue)/2+1))

    # Get size of R_int, T_int
    m = len(A) # get length of one side of A
    m = np.sqrt(m) # get square root, which is the length of the interior
    m = (m - 1) / 2 # get length of new iterior after restriction

    R_int = get_Rint(m, 2)
    T_int = get_Tint(m, 2)
    temp = np.dot(A, T_int)
    A = np.dot(R_int, temp) #RAT     

    # -----------------
    # Get error and use to improve u
    # -----------------
    e = v_sched_2d(np.zeros((len(residue), len(residue))), A, residue, 2.0 * dx, num_pre=num_pre, num_post=num_post, level=level-1, method=method)
    
    T = get_T(len(e), 2)
    # print [sum(T[i, :] for i in len(T[:,0]))]

    e = np.dot(T, e.reshape(len(e)**2))
    e = e.reshape((len(u), len(u)))

    u = u + e

    # -----------------
    # Relax on Au = rhs (num_post)
    # -----------------
    u = iterative_solver_2d(u, rhs, dx, num_iterations=num_post, method=method)

    return u

class MultiGridClass:
    ''' 
    Class to manage iterative PDE solvers with 
    multigrid approach

    Class Fields:
    x:  a numpy array that contains finest discretization 
        (contains end points)
    current_x: a numpy array that contains current discretization 
               for some grid level
    level: current grid level we are at (1 represents finest mesh or
           upper most grid level)
    u: numpy array that contains solution to laplacian 
    init_guess: initial guess
    domain: tuple that contains domain endpoints 
    bc: a tuple that contains BCs
    solver: a string that specifies the type of solver to be used
    f:   a function (non-homogenous term in u''(x) = f(x) 
    max_levels: the coarest mesh that can be used for given discretization


    '''
    # One Dimensional init method
    def __init__(self, x, U0, domain, f, bc=(0.0,0.0), solver='jacobi'):
        
        # array: grid discritization
        self.x = x

        # array: solution, U0 represents initial guess
        self.u = U0  

        # tuple: domain endpoints (x0, x1)
        self.domain = domain

        # tuple: Boundary conditions (alpha, beta)
        self.bc = bc

        # String: iterative method name ex: 'jacobi'
        self.solver = solver

        # function: non-homogenous term
        self.f = f

        # int: iteration counter
        self.iter_count = 0

        # make sure that the discretization is acceptable
        # we can only have discretizations of the form 
        # 2^j + 1, for j = [1, 2, ...]
        # [1, 3, 5, 9, 17, 33]
        self.max_level = np.log2(len(x)-1)
        assert(self.max_level % 1 == 0.0) 
        self.max_level = int(self.max_level) - 2

        # matrix is square and only includes interior points
        mat_size = len(x)-2   
        e = np.ones(mat_size)
        self.A = sparse.spdiags([e, -2*e, e], [-1,0,1], mat_size, mat_size).toarray()
        self.A = self.A / (x[1] - x[0])**2 

        self.plot_title = 'Initial Guess'

    # Two Dimensional init method
    def __init__(self, x, y, U0, domain, f, bc, solver='GS'):
        
        # array: grid discritization
        self.x = x
        self.y = y

        # array: solution, U0 represents initial guess
        self.u = U0  

        # tuple: domain endpoints (x0, x1)
        self.domain = domain

        # tuple: Boundary conditions (alpha, beta)
        self.bc = bc

        # String: iterative method name ex: 'jacobi'
        self.solver = solver

        # function: non-homogenous term
        self.f = f

        # int: iteration counter
        self.iter_count = 0

        # make sure that the discretization is acceptable
        # we can only have discretizations of the form 
        # 2^j + 1, for j = [1, 2, ...]
        # [1, 3, 5, 9, 17, 33]
        self.max_level = np.log2(len(x)-1)
        assert(self.max_level % 1 == 0.0) 
        self.max_level = int(self.max_level)


        # Construct A
        m = len(x) - 2
        e = np.ones(m)
        T = sparse.spdiags([e, -4 * e, e], [-1, 0, 1], m, m)
        S = sparse.spdiags([e, e], [-1, 1], m, m)
        I = sparse.eye(m)
        A = sparse.kron(I, T) + sparse.kron(S, I).toarray()
        self.A = A / ((x[1] - x[0])**2)
        self.plot_title = 'Initial Guess'


    def get_error(self):
        return np.linalg.norm(self.u - u_true(self.x), ord=2)

    def v_sched(self, num_pre=2, num_post=2):
        ''' Calls v_sched() defined above to approximate
        self.u using multigrid techniques with coarse grid 
        correction scheme.'''
       
        # used for plotting
        self.plot_title = 'MG'

        u = self.u.copy()
        A = self.A
        rhs = self.f(self.x)
        dx = self.x[1] - self.x[0]
        self.u = v_sched(u, A, rhs, dx, num_pre=num_pre, num_post=num_post, 
                            level=self.max_level, method=self.solver)

        self.iter_count = self.iter_count + num_pre + num_post

    def v_sched_2d(self, num_pre=2, num_post=2):
        ''' Calls v_sched() defined above to approximate
        self.u using multigrid techniques with coarse grid 
        correction scheme.'''
       
        # used for plotting
        self.plot_title = 'MG'

        u = self.u.copy()
        A = self.A
        X, Y = np.meshgrid(self.x, self.y)
        rhs = self.f(X, Y)
        dx = self.x[1] - self.x[0]
        self.u = v_sched_2d(u, A, rhs, dx, num_pre=num_pre, num_post=num_post, level=self.max_level, method=self.solver)

        self.iter_count = self.iter_count + num_pre + num_post

    def iterate_2d(self, num=2):
        ''' Calls v_sched() defined above to approximate
        self.u using multigrid techniques with coarse grid 
        correction scheme.'''
        
        # used for plotting
        self.plot_title = 'iterative'

        u = self.u.copy()
        X, Y = np.meshgrid(self.x, self.y)
        rhs = self.f(X, Y)        
        dx = self.x[1] - self.x[0]
        self.u = iterative_solver_2d(u, rhs, dx, num_iterations=num, method=self.solver)

        self.iter_count = self.iter_count + num

    def plot(self, plot_error=True):

        if plot_error:
            fig = plt.figure()
            axes = fig.add_subplot(1,2,2)
            axes.plot(self.x, abs(self.u (self.x)), label='$|U- u_{true}|$')
            axes.legend()

            axes = fig.add_subplot(1,2,1)

        else:
            fig = plt.figure()
            axes = fig.add_subplot(1,2,2)
        
        axes.plot(self.x, self.u, label=self.plot_title)
        axes.plot(self.x, u_true(self.x), label='$u_{true}$')        
        axes.legend()

    def plot_2d(self, u_true, plot_error=True):
        ''' Plot u(x) '''
        fig = plt.figure(figsize=(15,7))
        # fig.set_figwidth(fig.get_figwidth())
        axes = fig.add_subplot(1, 3, 1)
        plot = axes.pcolor(self.x, self.y, self.u, cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title(self.plot_title + ' - ' + self.solver)
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(1, 3, 2)
        X, Y = np.meshgrid(self.x, self.y)
        plot = axes.pcolor(self.x, self.y, u_true(X, Y), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("True Solution")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(1, 3, 3)
        X, Y = np.meshgrid(self.x, self.y)
        plot = axes.pcolor(self.x, self.y, abs(u_true(X, Y)-self.u), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("error")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        











