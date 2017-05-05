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

from getMatrices import get_T, get_R, get_Tint, get_Rint

def iterative_solver(u, rhs, dx, num_iterations=1, method='GS'):
        ''' Approximiating u system Au = rhs using an iterative method
        *input*
        u:  size total boundary, including endpoints, input is initial guess
        rhs: size total boundary, including endpoints, input should be unchanged

        *output*
        u: size unchanges, but closer approximation to the solution'''
        # Set omega for SOR
        omega = 2.0/3.0

        m  = len(rhs)
        for j in range(num_iterations):
            for i in range(1, m - 1):

                # interior jacobi iteration
                if self.solver=='GS':
                    u[i] = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])

                elif self.solver=='SOR':
                    u_gs = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])    
                    u[i] += omega * (u_gs - u[i])
        return u

def v_sched(u, A, rhs, dx, num_pre=2, num_post=2, num_down=1, num_up=1, method='GS'):
    ''' Approximiating u system Au = rhs using an iterative method
    *input*
    u:  size total boundary, including endpoints, input is initial guess
    rhs: size total boundary, including endpoints, input should be unchanged
    A: dimenssions of the interior of the problem

    *output*
    u: size unchanges, but closer approximation to the solution'''
        
    ## BASE CASE
    if len(u) <= 4:
        u[1:-1] = np.linalg.solve(A, rhs[1:-1])
        return u
    
    # -----------------
    # Relax on Au = rhs
    # -----------------

    u = iterative_solver(u, rhs, dx, num_iterations=num_pre, method='GS')
    
    # -----------------
    # Solve defect eq for residue
    # -----------------
    
    # Handle BC's
    residue = np.zeros(len(rhs))
    residue[1:-1] = rhs[1:-1] - np.dot(A, u[1:-1])
    R = get_R(len(residue))
    residue = np.dot(R, residue)

    T = get_Tint(len(A[::2]))
    A = nump.dot(R, np.dot(A, T)) #RAT 

    # -----------------
    # Get error and use to improve u
    # -----------------

    e, rhs_e, dx_e = v_sched(np.zeros(len(residue)), A, residue, 2*dx, num_pre=num_pre, num_post=num_post, \
                    num_down=num_down, num_up=num_up, method='GS')

    e = np.dot(T, e)

    u[1:-1] += e

    # -----------------
    # Restrict A, u, rhs
    # -----------------

    # Needed for A matricies when coming back up in second for loop
    A_list = [A]

    for i in xrange(num_down):
        
        # get T_int, R_int, R, 
        R = get_R(len(rhs))
        R_int = get_Rint(len(A))
        T_int = get_Tint(len(A[::2]))

        # Restrict u, rhs
        u = np.dot(R, u)
        rhs = np.dot(R, rhs)

        # Restrict A
        A = nump.dot(R_int, np.dot(A, T_int)) #RAT 
        A_list.append(A)

        new_u, rhs, dx = v_sched(new_u, A, new_rhs, 2*dx, num_pre=num_pre, num_post=num_post, \
                    num_down=num_down, num_up=num_up, method='GS')
    
    u = new_u.copy

    # -----------------
    # Relax on Au = rhs
    # -----------------
    
    for j in xrange(num_up):

        # get T 
        T = get_T(len(residue))

        # Interpolate u, rhs
        new_u = np.dot(T, u)
        new_rhs = np.dot(T, rhs)

        # Interpolate by getting the last A
        A = A_list.pop()

        u, rhs, dx = v_sched(new_u, A, new_rhs, .5 * dx, num_pre=num_pre, num_post=num_post, \
                    num_down=num_down, num_up=num_up, method='GS')
    
    return new_u, rhs, dx

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

    def __init__(self, x, U0, domain, f, bc=(0.0,0.0), solver='jacobi'):
        
        # array: grid discritization
        self.x = x

        self.current_x = x
        self.level = 1
        # self.y (to be added for 2d problems)

        # array: solution, U0 represents initial guess
        self.u = U0  
        self.init_guess = U0

        # tuple: domain endpoints (x0, x1)
        self.domain = domain

        # tuple: Boundary conditions (alpha, beta)
        self.bc = bc

        # String: iterative method name ex: 'jacobi'
        self.solver = solver

        # function: non-homogenous term
        self.f = f

        # make sure that the discretization is acceptable
        # we can only have discretizations of the form 
        # 2^j + 1, for j = [1, 2, ...]
        # [1, 3, 5, 9, 17, 33]
        self.max_levels = np.log2(len(x)-1)
        assert(self.max_levels % 1 == 0.0) 
        self.max_levels = int(self.max_levels)
        self.A_list = []

        # matrix is square and only includes interior points
        mat_size = len(x)-2   
        e = np.ones(mat_size)
        A = sparse.spdiags([e, -2*e, e], [-1,0,1], mat_size, mat_size).toarray()
        A /= (x[1] - x[0])**2 
        self.A_list.append(A)

        x_temp = x

        # determine all A matrices
        for i in xrange(self.max_levels-1):
             nextSize = .5 * (len(x_temp) - 1) - 1
             newT = get_Tint(nextSize)
             newR = get_Rint(nextSize) 

             # print "R shape: ", newR.shape
             # print "A shape: ", self.A_list[-1].shape
             # print "T shape: ", newT.shape

            # source: strang
             self.A_list.append(np.dot(newR, np.dot(self.A_list[-1], newT ) ))
             x_temp = x_temp[::2]


    def restrict(self):
        ''' Fine to coarse grid using matrix R for projection '''
        R = get_R(len(self.current_x[::2]))
        self.u = np.dot(R, self.u)
        self.current_x = self.current_x[::2]
        self.level = self.level + 1


    def interpolate(self):
        ''' Coarse to fine grid using matrix T for interpolation '''
        self.level = self.level -1
        T = get_T(len(self.current_x))
        self.u = np.dot(T, self.u)
        jump = 2 ** (self.level - 1)
        self.current_x = self.x[::jump]

    def iterative_solver(self, u, rhs=None, num_times=2):
        ''' 
        u in beginning is init guess
        Execute the interative solver to improve the solution u
        on current grid '''
        # omega = 2.0/3.0
        if rhs == None: 
            x_bc = self.current_x
            xx = x_bc[1:-1]
            dx = xx[1] - xx[0]
            m  = len(xx)
            for jj in range(num_times):
                for ii in range(1, m + 1):
                    # interior jacobi iteration
                    if self.solver=='jacobi':
                        u[ii] = 0.5 * (u[ii+1] + u[ii-1]) - self.f(x_bc[ii]) * dx**2 / 2.0
                    elif self.solver=='SOR':
                        omega = 2.0 / 3.0
                        u_gs = 0.5 * (u[ii-1] + u[ii+1] - dx**2 * self.f(x_bc[ii]))    
                        u[ii] += omega * (u_gs - u[ii])
        else: 
            x_bc = self.current_x
            xx = x_bc[1:-1]
            m = len(rhs)
            rhs = np.concatenate(( [0], rhs, [0] ))
            
            u = np.zeros((m+2))
            dx = xx[1] - xx[0]
            for jj in range(num_times):
                for ii in range(1,m + 1):
                    if self.solver=='jacobi':
                        u[ii] = 0.5 * (u[ii+1] + u[ii-1]) - rhs[ii] * dx**2 / 2.0
                    elif self.solver=='SOR':
                        # omega = 2.0 / 3.0
                        u_gs = 0.5 * (u[ii-1] + u[ii+1] - dx**2 * rhs[ii])
                        u[ii] += omega * (u_gs - u[ii])
            return u[1:-1]

    def plot_true(self):
        ''' Plot u(x) '''
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.current_x, self.u, label='$u$')
        axes.set_title(r"Multigrid Solution to $u''(x)=f(x)$")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$u(x)$")
        axes.legend(loc=2)
        plt.show()

    def get_error(self, u_true):
        return np.linalg.norm(self.u-u_true(self.current_x))

    def v_sched(self, num_pre=2, num_post=2, 
        num_down=1, num_up=1, u_true=None):
        # def print_error():
        #     if u_true != None:
        #         print 'Error with mg', self.get_error(u_true)

        # Relax num_pre times
        rhs = self.f(self.current_x)
        dx = self.current_x[1] - self.current_x[0]
        self.u = iterative_solver(self.u, rhs, dx, num_iterations=num_pre, method='GS')


        for i in xrange(num_down):
            self.restrict()
            self.iterative_solver(num_times=num_pre)
            print_error()

        for j in xrange(num_up):
            self.interpolate()
            self.iterative_solver(num_times=num_post)
            print_error()
    def solveCGC(self, num_pre = 2, num_post=2, 
                    num_down=1, num_up=1, u_true=None):
        ''' 
        Coarse Grid Correction Scheme 
        num_pre:  number of times pre-smoothing
        num_post: number of times post-smoothing
        num_down: how many times we restrict down
        num_up:   how many times we interpolate up
        '''
        def print_error():
            if u_true != None:
                print 'Error with mg', self.get_error(u_true)
        
        # relax num_pre times on A^h u^h = f^h
        self.iterative_solver(self.u, num_times = num_pre)
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.current_x, self.u, label='$u$')
        axes.set_title(r"2 sweeps Jacobi to $u''(x)=0$")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$u(x)$")
        axes.legend(loc=2)
        plt.savefig("../plots/mg_pre_2sweeps_jacobi.pdf")

        # number of interior points in mesh
        numInt = len(self.current_x) - 2
        R = get_Rint(numInt, dim=1)
        #R = .5*get_Tint( (numInt-1) /2 ).T

        intPoints = self.current_x[1:-1]
        res = self.f(intPoints) - np.dot(self.A_list[0], self.u[1:-1])
        # restrict the residual to coarser mesh
        res = np.dot(R, res) 

        # solve error/residual equation on coarser mesh
        error = np.linalg.solve(self.A_list[1], res)
        
        # correct fine grid approximation
        T = get_Tint(numInt=len(self.current_x[::2])-2)
        self.u[1:-1] = self.u[1:-1] + np.dot(T, error)

        # relax num_post times on A^h u^h = f^h
        self.iterative_solver(self.u, num_times = num_post)

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.current_x, self.u, label='$u$')
        axes.set_title(r"Multigrid Solution $u''(x)=0$")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$u(x)$")
        axes.legend(loc=2)
        plt.savefig("../plots/mg_post.pdf")

    def test(self, num_pre = 2, num_post=2, 
            num_down=1, num_up=1, u_true=None):
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        #axes.plot(self.current_x, self.u,  'r-', label='guess')

        # pre-smoothing relaxation
        self.iterative_solver(self.u, num_times = num_pre)

        # number of interior points in mesh
        numInt = len(self.current_x) - 2

        T = get_Tint(numInt=len(self.current_x[::2])-2)
        R = get_Rint(numInt, dim=1)

        intPoints = self.current_x[1:-1]
        # compute residual and restrict it to coarser mesh
        res = self.f(intPoints) - np.dot(self.A_list[0], self.u[1:-1])
        res = np.dot(R, res) 

        # relax three times on Ae = r, note we do not solve system here
        error = self.iterative_solver(res, rhs=res, num_times=3)

        #correct fine grid approx.
        self.u[1:-1] = self.u[1:-1] + np.dot(T, error)

        # relax three more times with new approximation
        self.iterative_solver(self.u, num_times = num_pre)
        res = self.f(intPoints) - np.dot(self.A_list[0], self.u[1:-1])
        res = np.dot(R, res) # coarsen residual again

        # relax three times on Ae = r, note we do not solve system here
        error = self.iterative_solver(res, rhs=res, num_times=3)

        self.u[1:-1] = self.u[1:-1] + np.dot(T, error)
        
        axes.plot(self.current_x, self.u, label='$u$')
        axes.set_title(r"good Multigrid Solution $u''(x)=0$")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$u(x)$")
        axes.legend(loc=2)
        plt.savefig("../plots/mg_post_error_smooth.pdf")





















