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
####

import numpy
from matplotlib import pyplot as plt
import scipy.sparse as sparse

class MultiGridClass(self):
    ''' Class to manage iterative PDE solvers with 
        multigrid approach '''

    def __init__(self, x, U0, domain, bc=(0.0,0.0), solver='jacobi'):
        # array: grid descritization
        self.x = x
        # self.y (to be added for 2d problems)

        # array: solution
        self.u = U0

        # tuple: domain endpoints (x0, x1)
        self.domain = domain

        # tuple: Boundary conditions (alpha, beta)
        self.bc = bc

        # String: iterative method name ex: 'jacobi'
        self.solver = solver

    def restrict(self):
        ''' Fine to coarse grid using matrix R for projection '''
        raise NotYetImplemented

    def interpolate(self):
        ''' Coarse to fine grid using matrix T for interpolation '''
        raise NotYetImplemented

    def iterative_solver(self, num_times=1):
        ''' Execute the interative solver to improve the solution u
        on current grid '''
        raise NotYetImplemented

    def plot(self):
        ''' Plot u(x) '''
        raise NotYetImplemented




