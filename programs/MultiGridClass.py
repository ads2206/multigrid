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
        for i in xrange(self.max_levels):
             
             nextSize = .5 * (len(x_temp) - 1) - 1
             newR = get_Rint(nextSize) 
             newT = get_Tint(nextSize)
             #print "R shape: ", newR.shape
             #print "A shape: ", self.A_list[-1].shape
             #print "T shape: ", newT.shape
             # do not make a A matrix when we have only one interior point
             if self.A_list[-1].shape == (1,1):
                break
            # source: strang
             self.A_list.append(np.dot(newR, np.dot(self.A_list[-1], newT ) ))
             x_temp = x_temp[::2]

        # for a in self.A_list:
        #     print a


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

    def iterative_solver(self, num_times=2):
        ''' 
        Execute the interative solver to improve the solution u
        on current grid '''
        x_bc = self.current_x
        xx = x_bc[1:-1]
        dx = xx[1] - xx[0]
        m  = len(xx)
        for jj in range(num_times):
            for ii in range(1, m + 1):
                self.u[ii] = 0.5 * (self.u[ii+1] + self.u[ii-1]) - self.f(x_bc[ii]) * dx**2 / 2.0

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

    def v_sched(self, num_pre = 2, num_post=2, 
        num_down=1, num_up=1, u_true=None):
        def print_error():
            if u_true != None:
                print 'Error with mg', self.get_error(u_true)

        self.iterative_solver()
        print_error()

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
        self.iterative_solver(num_times = num_pre)
        
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
        #R = get_Rint(numInt, dim=1)
        R = .5*get_Tint( (numInt-1) /2 ).T
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
        self.iterative_solver(num_times = num_post)

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.current_x, self.u, label='$u$')
        axes.set_title(r"Multigrid Solution $u''(x)=0$")
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$u(x)$")
        axes.legend(loc=2)
        plt.savefig("../plots/mg_post.pdf")









