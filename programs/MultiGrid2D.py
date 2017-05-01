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
import scipy.sparse as sparse

class MultiGrid2D:
    ''' Class to manage iterative PDE solvers with 
        multigrid approach '''

    def __init__(self, x, y, U0, domain, f, bc, u_true, name, solver='jacobi'):
        # array: grid descritization
        self.name = name
        self.x = x
        self.y = y
        self.current_x = x
        self.current_y = y
        # manually insert maximum level (coarsest mesh)
        self.max_levels = 3


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

        ### NEEDS WORK FOR RELAXING ERROR
        # Some matrices to keep around:
        # self.max_levels = np.log2(len(x)-1)
        # assert(self.max_levels % 1 == 0)
        # self.max_levels = int(self.max_levels)
        self.A_list = []
        # Store matrix A
        mat_size = len(x)**2
        e = np.ones(mat_size)
        A = sparse.spdiags([e, -2*e, e], [-1,0,1], mat_size, mat_size).toarray()
        A /= (x[1] - x[0])**2 
        self.A_list.append(A)
        # print A
        for i in xrange(self.max_levels):
            A = self.A_list[-1]
            self.A_list.append(np.dot(get_R(int(np.sqrt(len(A)))/2 + 1, 2), np.dot(A, get_T(int(np.sqrt(len(A))) /2  + 1, 2))))

    def set_boundary(self):
        self.u[0, :] = self.bc[2](self.current_x)
        self.u[-1, :] = self.bc[3](self.current_x)
        self.u[:, 0] = self.bc[0](self.current_y)
        self.u[:, -1] = self.bc[1](self.current_y)

    def restrict(self):
        ''' Fine to coarse grid using matrix R for projection '''
        print 'number of points in x direction', len(self.current_x)
        R = get_R(len(self.current_x[::2]), 2)
        self.u = np.dot(R, self.u.reshape(len(self.u)**2))
        self.current_x = self.current_x[::2]
        self.current_y = self.current_y[::2]
        self.level += 1
        square = int(np.sqrt(len(self.u)))
        self.u = self.u.reshape((square,square))
        self.set_boundary()
        # self.u[0, :] = self.bc[2](self.current_x)
        # self.u[-1, :] = self.bc[3](self.current_x)
        # self.u[:, 0] = self.bc[0](self.current_y)
        # self.u[:, -1] = self.bc[1](self.current_y)
        print 'number of points in x direction', len(self.current_x)

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
        self.set_boundary()

        # self.u[0, :] = self.bc[2](self.current_x)
        # self.u[-1, :] = self.bc[3](self.current_x)
        # self.u[:, 0] = self.bc[0](self.current_y)
        # self.u[:, -1] = self.bc[1](self.current_y)

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
            # self.set_boundary()

    def plot_error(self):
        ''' plot the error agains the interation count.'''
        fig = plt.figure()
        fig.set_figwidth(fig.get_figwidth())
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.error)

    def plot(self, u_true):
        ''' Plot u(x) '''
        fig = plt.figure()
        fig.set_figwidth(fig.get_figwidth())
        axes = fig.add_subplot(1, 3, 1)
        plot = axes.pcolor(self.current_x, self.current_y, self.u, cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("Computed Solution - MG")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(1, 3, 2)
        X, Y = np.meshgrid(self.current_x, self.current_y)
        plot = axes.pcolor(self.current_x, self.current_y, u_true(X, Y), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("True Solution")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(1, 3, 3)
        X, Y = np.meshgrid(self.current_x, self.current_y)
        plot = axes.pcolor(self.current_x, self.current_y, abs(u_true(X, Y)-self.u), cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("error")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        # fig.set_figwidth(fig.get_figwidth())
        # axes = fig.add_subplot(2, 3, 4)
        # plot = axes.pcolor(self.current_x, self.current_y, u_test, cmap=plt.get_cmap("Blues"))
        # fig.colorbar(plot, label="$U$")
        # axes.set_title("Computed Solution")
        # axes.set_xlabel("x")
        # axes.set_ylabel("y")

        # axes = fig.add_subplot(2, 3, 5)
        # X, Y = np.meshgrid(self.current_x, self.current_y)
        # plot = axes.pcolor(self.current_x, self.current_y, u_true(X, Y), cmap=plt.get_cmap("Blues"))
        # fig.colorbar(plot, label="$U$")
        # axes.set_title("True Solution")
        # axes.set_xlabel("x")
        # axes.set_ylabel("y")

        # axes = fig.add_subplot(2, 3, 6)
        # X, Y = np.meshgrid(self.current_x, self.current_y)
        # plot = axes.pcolor(self.current_x, self.current_y, abs(u_true(X, Y)-u_test), cmap=plt.get_cmap("Blues"))
        # fig.colorbar(plot, label="$U$")
        # axes.set_title("Error")
        # axes.set_xlabel("x")
        # axes.set_ylabel("y")

        plt.savefig("%s.pdf" % self.name)
        #plt.show()

    def get_error(self):
        X, Y = np.meshgrid(self.current_x, self.current_y)

        return np.linalg.norm(self.u-self.u_true(X,Y), ord=np.infty)

    def v_sched(self, num_down=2, num_up=2, u_true=None):
        def print_error():
            if u_true != None:
                print 'Error with mg', self.get_error(u_true)

        # self.iterative_solver() 
        self.step()
        print_error()

        for i in xrange(num_down):
            self.restrict()
            # self.iterative_solver() 
            self.step()
            print_error()

            # solve linear system exactly
            # if self.level == self.max_levels:
            #     mat_size = len(self.current_x) - 2
            #     print mat_size
            #     e = np.ones(mat_size)
            #     A = sparse.spdiags([e, -2.*e, e], [-1,0,1], mat_size**2, mat_size**2).toarray()
            #     A /= (self.current_x[1] - self.current_x[0])**2
            #     xx, yy = np.meshgrid(self.current_x, self.current_y)
            #     rhs = self.f(xx, yy)[1:-1,1:-1].reshape(mat_size**2)

            #     tmp = np.linalg.solve(A, rhs)
            #     self.u[1:-1,1:-1] = tmp.reshape(mat_size, mat_size)



        for j in xrange(num_up):
            self.interpolate()
            # self.iterative_solver() 
            self.step()
            print_error()

    def step(self, pre=1, post=1):
        ''' one level of relaxation on u'''
        X, Y = np.meshgrid(self.current_x, self.current_y)
        
        for ii in xrange(pre): self.iterative_solver()
        
        d = self.f(X, Y) - np.dot(self.A_list[self.level-1], self.u.reshape(len(self.u)**2)).reshape((len(self.u), len(self.u)))
        R = get_R(len(self.current_x[::2]), 2)
        d = np.dot(R, d.reshape(len(d)**2))
        # d = d.reshape(int(np.sqrt(len(d))), int(np.sqrt(len(d))))

        # Solve Av = d
        # print self.A_list[self.level].shape
        # print d.reshape(len(d)**2).shape
        v = np.linalg.solve(self.A_list[self.level], d)
        T = get_T(len(self.current_x[::2]), 2)
        v = np.dot(T, v)
        v = v.reshape(int(np.sqrt(len(v))), int(np.sqrt(len(v))))

        # improve u
        self.u = self.u + v
        self.set_boundary()

        for jj in xrange(post): self.iterative_solver()


    def test(self):
        self.u[:,:] = 2.1





