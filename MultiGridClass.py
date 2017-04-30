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
from getMatrices import get_T, get_R

class MultiGridClass(self):
    ''' Class to manage iterative PDE solvers with 
        multigrid approach '''

    def __init__(self, x, U0, domain, f, bc=(0.0,0.0), solver='jacobi'):
        # array: grid descritization
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

    def restrict(self):
        ''' Fine to coarse grid using matrix R for projection '''
        R = get_R(len(self.x))
        self.u = np.dot(R, self.u)
        self.current_x = self.current_x[::2]
        self.level += 1


    def interpolate(self):
        ''' Coarse to fine grid using matrix T for interpolation '''
        self.level -= 1
        T = get_T(len(self.x))
        self.u = np.dot(T, self.u)
        jump = 2 ** (self.level - 1)
        self.current_x = self.x[::jump]

    def iterative_solver(self, num_times=1):
        ''' Execute the interative solver to improve the solution u
        on current grid '''
        x_bc = self.current_x
        x = x_bc[1:-1]
        dx = x[1] - x[0]
        m  = len(x)
        for j in range(num_times):
            for i in range(1, m + 1):
                self.u[i] = 0.5 * (self.u[i+1] + self.u[i-1]) - f(x_bc[i]) * dx**2 / 2.0


    def plot(self):
        ''' Plot u(x) '''
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.current_x, self.u)
            axes.set_title("Title needed")
        axes.set_xlabel("x")
        axes.set_ylabel("u(x)")
        axes.legend(loc=2)
        plt.show()



