############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# MultiGrid Classes
# MultiGrid1d and MultiGrid2d
####

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from mg_module import iterative_solver, iterative_solver_2d
from mg_module import v_sched, v_sched_2d, fmg, fmg_2d

class MultiGrid1d:
    ''' 
    Class to solve the one dimensional non-homogeneous Laplacian
    with Direchlet BC's of zero using a multigrid approach

    Class Fields:
    x:  a numpy array that contains finest discretization 
        (contains end points)
    
    u: numpy array that contains solution to PDE

    U0: initial guess
    
    domain: tuple that contains domain endpoints 
    
    solver: a string that specifies the iterative solver to be used
    
    f:   a function (non-homogenous term in u''(x) = f(x) 
    
    max_levels: the coarest mesh that can be used for given discretization
    '''

    def __init__(self, x, U0, domain, f, solver='GS'):
        ''' Constructor method '''
        
        # array: grid discritization
        self.x = x

        # array: solution, U0 represents initial guess
        self.u = U0 

        # tuple: domain endpoints (x0, x1)
        self.domain = domain

        # String: iterative method name ex: 'GS' for Gauss-Seidel
        self.solver = solver

        # function: non-homogenous term
        self.f = f

        # int: iteration counter
        self.iter_count = 0
        self.plot_title = 'Initial Guess'

        # make sure that the discretization is acceptable
        # we can only have discretizations of the form 
        # 2^j + 1, for j = [1, 2, ...]
        # [1, 3, 5, 9, 17, 33]
        self.max_level = np.log2(len(x)-1)
        assert(self.max_level % 1 == 0.0) 
        self.max_level = int(self.max_level)

        # A matrix is square and only includes interior points
        mat_size = len(x)-2   
        e = np.ones(mat_size)
        self.A = sparse.spdiags([e, -2*e, e], [-1,0,1], mat_size, mat_size).toarray()
        self.A = self.A / (x[1] - x[0])**2 



    def get_error(self, u_true, order=2):
        ''' Returns error: u - u_true '''
        return np.linalg.norm(self.u-u_true(self.x), ord=order)

    def v_sched(self, num_pre=2, num_post=2):
        ''' Calls v_sched() imported above to approximate
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
    
    def fmg(self, num_pre=2, num_post=2):
        ''' Calls full_multi_grid() imported above to run the full 
        multigrid cycle '''

        max_level = self.max_level
        u = self.u.copy()
        A = self.A
        rhs = self.f(self.x)
        dx = self.x[1] - self.x[0]

        self.u = fmg(u, A, rhs, dx, self.max_level, self.domain,
                            num_pre=num_pre, num_post=num_post, method=self.solver)
    
        self.plot_title='FMG'
        self.iter_count = self.iter_count + num_pre + num_post


    def iterate(self, num=2):
        ''' Calls iterate_solver() imported above to approximate
        self.u '''
        
        # used for plotting
        self.plot_title = 'iterative'

        u = self.u.copy()
        rhs = self.f(self.x)
        dx = self.x[1] - self.x[0]
        self.u = iterative_solver(u, rhs, dx, num_iterations=num, method=self.solver)

        self.iter_count = self.iter_count + num

    def plot(self, u_true, plot_error=True):
        ''' Plot u and u_true (and error) '''
        if plot_error:
            fig = plt.figure()
            axes = fig.add_subplot(1,2,2)
            axes.plot(self.x, abs(self.u - u_true(self.x)), label='$|U- u_{true}|$')
            axes.legend()
            axes.set_title('Error')

            axes = fig.add_subplot(1,2,1)

        else:
            fig = plt.figure()
            axes = fig.add_subplot(1,1,1)
        
        axes.set_title(self.plot_title + ' - ' + self.solver)
        axes.plot(self.x, self.u, label=self.plot_title)
        axes.plot(self.x, u_true(self.x), label='$u_{true}$')        
        axes.legend()
        


class MultiGrid2d:
    ''' 
    Class to solve the two dimensional non-homogeneous Laplacian
    with Direchlet BC's of zero using a multigrid approach

    Class Fields:
    x, y:  two numpy arrays that contains finest discretization 
        (contains end points)
    
    u: numpy array that contains solution to PDE

    U0: initial guess
    
    domain: tuple that contains domain endpoints 
    
    solver: a string that specifies the iterative solver to be used
    
    f:   a function, non-homogenous term in u_xx(x, y) + u_yy(x, y) = f(x, y) 
    
    max_levels: the coarest mesh that can be used for given discretization
    '''

    def __init__(self, x, y, U0, domain, f, solver='GS'):
        ''' Constructor method '''
        
        # array: grid discritization
        self.x = x
        self.y = y

        # array: solution, U0 represents initial guess
        self.u = U0  

        # tuple: domain endpoints (x0, x1)
        self.domain = domain

        # String: iterative method name ex: 'jacobi'
        self.solver = solver

        # function: non-homogenous term
        self.f = f

        # int: iteration counter
        self.iter_count = 0
        self.plot_title = 'Initial Guess'

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
        

    def get_error(self, u_true):
        ''' Returns error: u - u_true '''
        X, Y = np.meshgrid(self.x, self.y)
        return np.max(abs(u_true(X, Y)-self.u))
        # return np.linalg.norm(self.u - u_true(X, Y), ord=2)

    def v_sched(self, num_pre=2, num_post=2):
        ''' Calls v_sched_2d() imported above to approximate
        self.u using multigrid techniques with coarse grid 
        correction scheme.'''
       
        # used for plotting
        self.plot_title = 'MG'

        u = self.u.copy()
        A = self.A
        X, Y = np.meshgrid(self.x, self.y)
        rhs = self.f(X, Y)
        dx = self.x[1] - self.x[0]
        self.u = v_sched_2d(u, A, rhs, dx, num_pre=num_pre, num_post=num_post, 
                                level=self.max_level, method=self.solver)

        self.iter_count = self.iter_count + num_pre + num_post

    def fmg(self, num_pre=2, num_post=2):
        ''' Calls fmg_2d() imported above to approximate
        self.u using the multigrid technique of a Full 
        Multi Grid Cycle.'''
       
        # used for plotting
        self.plot_title = 'FMG'

        u = self.u.copy()
        A = self.A
        X, Y = np.meshgrid(self.x, self.y)
        rhs = self.f(X, Y)
        dx = self.x[1] - self.x[0]
        self.u = fmg_2d(u, A, rhs, dx, self.max_level, self.domain, num_pre=num_pre, 
                        num_post=num_post, method=self.solver)

        self.iter_count = self.iter_count + num_pre + num_post

    def iterate(self, num=2):
        ''' Calls iterative_solver_2d() imported above to approximate
        self.u using multigrid techniques with coarse grid 
        correction scheme.'''
        
        # used for plotting
        self.plot_title = 'iterative'

        u = self.u.copy()
        X, Y = np.meshgrid(self.x, self.y)
        rhs = self.f(X, Y)        
        dx = self.x[1] - self.x[0]
        self.u = iterative_solver_2d(u, rhs, dx, num_iterations=num, 
                                        method=self.solver)

        self.iter_count = self.iter_count + num

    def plot(self, u_true):
        ''' Plot u(x) with the true solution and the error'''

        fig = plt.figure(figsize=(15,7))
        axes = fig.add_subplot(1, 3, 1)
        plot = axes.pcolor(self.x, self.y, self.u, cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title(self.plot_title + ' - ' + self.solver)
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(1, 3, 2)
        X, Y = np.meshgrid(self.x, self.y)
        plot = axes.pcolor(self.x, self.y, u_true(X, Y), 
                            cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("True Solution")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes = fig.add_subplot(1, 3, 3)
        X, Y = np.meshgrid(self.x, self.y)
        plot = axes.pcolor(self.x, self.y, abs(u_true(X, Y)-self.u), 
                            cmap=plt.get_cmap("Blues"))
        fig.colorbar(plot, label="$U$")
        axes.set_title("error")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
 
