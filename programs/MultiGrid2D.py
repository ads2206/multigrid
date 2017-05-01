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

    def __init__(self, x, y, U0, domain, f, bc, u_true, solver='jacobi'):
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

        # Error time!
        # list of lists[ [iteration, error], ... ]
        self.error = []
        self.counter = 0    
        self.u_true = u_true

    def restrict(self):
        ''' Fine to coarse grid using matrix R for projection '''
        R = get_R(len(self.current_x[::2]), 2)
        self.u = np.dot(R, self.u.reshape(len(self.u)**2))
        self.current_x = self.current_x[::2]
        self.current_y = self.current_y[::2]
        self.level += 1
        square = int(np.sqrt(len(self.u)))
        self.u = self.u.reshape((square,square))
        self.u[0, :] = self.bc[2](self.current_x)
        self.u[-1, :] = self.bc[3](self.current_x)
        self.u[:, 0] = self.bc[0](self.current_y)
        self.u[:, -1] = self.bc[1](self.current_y)

    def interpolate(self):
        ''' Coarse to fine grid using matrix T for interpolation '''
        self.level -= 1
        T = get_T(len(self.current_x), 2)
        self.u = np.dot(T, self.u.reshape(len(self.u)**2))
        jump = 2 ** (self.level - 1)
        self.current_x = self.x[::jump]
        self.current_y = self.y[::jump]
        square = int(np.sqrt(len(self.u)))
        self.u = self.u.reshape((square,square))
        self.u[0, :] = self.bc[2](self.current_x)
        self.u[-1, :] = self.bc[3](self.current_x)
        self.u[:, 0] = self.bc[0](self.current_y)
        self.u[:, -1] = self.bc[1](self.current_y)

    def iterative_solver(self, num_times=20):
        ''' Execute the interative solver to improve the solution u
        on current grid '''
        x_bc = self.current_x
        x = x_bc[1:-1]
        dx = x_bc[1] - x_bc[0]
        m  = len(x)
        for k in range(num_times):
            for i in range(1, m + 1):
                for j in range(1, m + 1):
                    self.u[i, j] = 0.25 * (self.u[i+1, j] + self.u[i-1, j] + self.u[i, j-1] + self.u[i, j+1]) - self.f(self.current_x[i], self.current_y[j]) * dx**2 / 4.0
            self.counter += 1
            self.error.append(self.get_error())

    def plot_error(self):
        ''' plot the error agains the interation count.'''
        fig = plt.figure()
        fig.set_figwidth(fig.get_figwidth())
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.error)

    def plot(self, u_true, u_test):
        ''' Plot u(x) '''
        fig = plt.figure()
        fig.set_figwidth(fig.get_figwidth())
        axes = fig.add_subplot(2, 3, 1)
        plot = axes.pcolor(self.current_x, self.current_y, self.u, cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("Computed Solution - MG")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(2, 3, 2)
        X, Y = np.meshgrid(self.current_x, self.current_y)
        plot = axes.pcolor(self.current_x, self.current_y, u_true(X, Y), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("True Solution")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(2, 3, 3)
        X, Y = np.meshgrid(self.current_x, self.current_y)
        plot = axes.pcolor(self.current_x, self.current_y, abs(u_true(X, Y)-self.u), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("error")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        fig.set_figwidth(fig.get_figwidth())
        axes = fig.add_subplot(2, 3, 4)
        plot = axes.pcolor(self.current_x, self.current_y, u_test, cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("Computed Solution")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(2, 3, 5)
        X, Y = np.meshgrid(self.current_x, self.current_y)
        plot = axes.pcolor(self.current_x, self.current_y, u_true(X, Y), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("True Solution")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(2, 3, 6)
        X, Y = np.meshgrid(self.current_x, self.current_y)
        plot = axes.pcolor(self.current_x, self.current_y, abs(u_true(X, Y)-u_test), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("Error")
        axes.set_xlabel("x")
        axes.set_ylabel("y")


        plt.show()

    def get_error(self):
        X, Y = np.meshgrid(self.current_x, self.current_y)

        return np.linalg.norm(self.u-self.u_true(X,Y), ord=np.infty)

    def v_sched(self, num_down=3, num_up=3, u_true=None):
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


    def test(self):
        self.u[:,:] = 2.1





