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
# 2D version
####
import numpy as np
from matplotlib import pyplot as plt
from getMatrices import get_T, get_R

class MultiGrid2D:
    ''' Class to manage iterative PDE solvers with 
        multigrid approach '''

    def __init__(self, x, y, U0, domain, f, bc, solver='jacobi'):
        # array: grid descritization
        self.x = x
        self.y = y
        self.current_x = x
        self.current_y = y

        self.level = 1
        # self.y (to be added for 2d problems)

        # 2D array: solution
        self.u = U0
        self.u[0, :] = bc[2](x)
        self.u[-1, :] = bc[3](x)
        self.u[:, 0] = bc[0](y)
        self.u[:, -1] = bc[1](y)

        # tuple: domain endpoints (x0, x1, y0, y1)
        self.domain = domain

        # tuple of functions: Boundary conditions (alpha_x(y), beta_x(y), alpha_y(x), beta_y(x))
        self.bc = bc

        # String: iterative method name ex: 'jacobi'
        self.solver = solver

        # function: non-homogenous term
        self.f = f

    def restrict(self):
        ''' Fine to coarse grid using matrix R for projection '''
        R = get_R(len(self.current_x[::2]), 2)
        self.u = np.dot(R, self.u.reshape(len(self.u)**2))
        self.current_x = self.current_x[::2]
        self.current_y = self.current_y[::2]
        self.level += 1
        square = np.sqrt(len(self.u))
        self.u = self.u.reshape((square,square))


    def interpolate(self):
        ''' Coarse to fine grid using matrix T for interpolation '''
        self.level -= 1
        T = get_T(len(self.current_x), 2)
        self.u = np.dot(T, self.u.reshape(len(self.u)**2))
        jump = 2 ** (self.level - 1)
        self.current_x = self.x[::jump]
        self.current_y = self.y[::jump]
        square = np.sqrt(len(self.u))
        self.u = self.u.reshape((square,square))

    def iterative_solver(self, num_times=2):
        ''' Execute the interative solver to improve the solution u
        on current grid '''
        x_bc = self.current_x
        x = x_bc[1:-1]
        dx = x[1] - x[0]
        m  = len(x)
        for k in range(num_times):
            for i in range(1, m + 1):
                for j in range(1, m + 1):
                    self.u[i, j] = 0.25 * (self.u[i+1, j] + self.u[i-1, j] + self.u[i, j-1] + self.u[i, j+1]) - self.f(self.current_x[i], self.current_y[j]) * dx**2 / 4.0

    def plot(self, u_true=None):
        ''' Plot u(x) '''
        fig = plt.figure()
        fig.set_figwidth(fig.get_figwidth())
        axes = fig.add_subplot(1, 1, 1)
        plot = axes.pcolor(self.current_x, self.current_y, self.u, cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("Computed Solution")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        if u_true:
            fig = plt.figure()
            fig.set_figwidth(fig.get_figwidth())
            axes = fig.add_subplot(1, 1, 1)
            X, Y = np.meshgrid(self.current_x, self.current_y)
            plot = axes.pcolor(self.current_x, self.current_y, u_true(X, Y), cmap=plt.get_cmap("Blues"))
            fig.colorbar(plot, label="$U$")
            axes.set_title("Computed Solution")
            axes.set_xlabel("x")
            axes.set_ylabel("y")
        plt.show()

    def get_error(self, u_true):
        X, Y = np.meshgrid(self.current_x, self.current_y)

        return np.linalg.norm(self.u-u_true(X,Y))

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






