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

import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse

from getMatrices import get_T, get_R

class MultiGridClass:
    ''' 
    Class to manage iterative PDE solvers with 
    multigrid approach
    '''
    def __init__(self, x, U0, domain, f, bc=(0.0,0.0), solver='jacobi'):
        # array: grid discritization
        self.x = x
        self.current_x = x
        self.level = 1
        # self.y (to be added for 2d problems)

        # array: solution
        self.u = U0

        # tuple: domain endpoints (x0, x1)
        self.domain = domain

        # tuple: Boundary conditions (alpha, beta)
        self.bc = bc

        # String: iterative method name ex: 'jacobi'
        self.solver = solver

        # function: non-homogenous term
        self.f = f

        ### NEEDS WORK FOR RELAXING ERROR
        
        # Some matrices to keep around:

        # make sure that the discretization is acceptable
        # we can only have discretizations of the form 
        # 2^j + 1, for j = [1, 2, ...]
        # [1, 3, 5, 9, 17, 33]
        self.max_levels = np.log2(len(x)-1)
        assert(self.max_levels % 1 == 0.0) 
        self.max_levels = int(self.max_levels)
        A_list = []

        # Store matrix A
        # matrix is square and only includes interior points
        mat_size = len(x)-2
        e = np.ones(mat_size)
        A = sparse.spdiags([e, -2*e, e], [-1,0,1], mat_size, mat_size).toarray()
        A /= (x[1] - x[0])**2 
        A_list.append(A)

        for i in xrange(self.max_levels):
             # source: strang
             print(len(A_list[-1]))
             nextSize = .5 * (len(A_list[-1]) - 1) + 1
             print(nextSize)
             newR = get_R(nextSize) 
             newT = get_T(nextSize)
             print(np.dot(A_list[-1], newT ))
             A_list.append(np.dot(newR, np.dot(A_list[-1], newT ) ))

        for a in A_list:
            print a


    def restrict(self):
        ''' Fine to coarse grid using matrix R for projection '''
        R = get_R(len(self.current_x[::2]))
        self.u = np.dot(R, self.u)
        self.current_x = self.current_x[::2]
        self.level += 1


    def interpolate(self):
        ''' Coarse to fine grid using matrix T for interpolation '''
        self.level -= 1
        T = get_T(len(self.current_x))
        self.u = np.dot(T, self.u)
        jump = 2 ** (self.level - 1)
        self.current_x = self.x[::jump]

    def iterative_solver(self, num_times=2):
        ''' Execute the interative solver to improve the solution u
        on current grid '''
        x_bc = self.current_x
        x = x_bc[1:-1]
        dx = x[1] - x[0]
        m  = len(x)
        for j in range(num_times):
            for i in range(1, m + 1):
                self.u[i] = 0.5 * (self.u[i+1] + self.u[i-1]) - self.f(x_bc[i]) * dx**2 / 2.0

    def plot(self):
        ''' Plot u(x) '''
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.current_x, self.u, label='u')
        axes.set_title("Title needed")
        axes.set_xlabel("x")
        axes.set_ylabel("u(x)")
        axes.legend(loc=2)
        plt.show()

    def get_error(self, u_true):
        return np.linalg.norm(self.u-u_true(self.current_x))

    def v_sched(self, num_down=2, num_up=2, u_true=None):
        def print_error():
            if u_true != None:
                print 'Error with mg', self.get_error(u_true)

        self.iterative_solver()
        print_error()

        for i in xrange(num_down):
            self.restrict()
            self.iterative_solver()
            print_error()

        for j in xrange(num_up):
            self.interpolate()
            self.iterative_solver()
            print_error()






