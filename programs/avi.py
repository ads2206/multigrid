############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# v_sched function
# (one dimensional for now.)
####

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from MultiGridClass import MultiGridClass as MGC


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
                if method=='GS':
                    u[i] = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])

                elif method=='SOR':
                    u_gs = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])    
                    u[i] += omega * (u_gs - u[i])
        return u

def v_sched(u, A, rhs, dx, num_pre=2, num_post=2, num_down=2, num_up=2, method='GS'):
    ''' Approximiating u system Au = rhs using an iterative method
    *input*
    u:  size total boundary, including endpoints, input is initial guess
    rhs: size total boundary, including endpoints, input should be unchanged
    A: dimenssions of the interior of the problem

    *output*
    u: size unchanges, but closer approximation to the solution'''
        
    print 'hey'
    ## BASE CASE
    if len(u) <= 8:
        print len(u), len(A), len(rhs)
        u[1:-1] = np.linalg.solve(A, rhs[1:-1])
        return u, rhs, dx
    
    # Needed for A matricies when coming back up in second for loop
    A_list = [A]
    
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
    R = get_R(len(residue) / 2 + 1 )
    residue = np.dot(R, residue)

    R_int = get_Rint(len(A[::2]) - 1)
    T_int = get_Tint(len(A[::2]) - 1)
    A = np.dot(R_int, np.dot(A, T_int)) #RAT 

    # -----------------
    # Get error and use to improve u
    # -----------------
    print 'line92'
    e, rhs_e, dx_e = v_sched(np.zeros(len(residue)), A, residue, 2*dx, num_pre=num_pre, num_post=num_post, num_down=num_down, num_up=num_up, method='GS')
    
    T = get_T(len(e))
    e = np.dot(T, e)

    u += e

    # -----------------
    # Restrict A, u, rhs
    # -----------------

    for i in xrange(num_down):
        
        # get T_int, R_int, R, 
        R = get_R(len(rhs[::2]))
        R_int = get_Rint(len(A[::2]) - 1)
        T_int = get_Tint(len(A[::2]) - 1)

        # Restrict u, rhs
        u = np.dot(R, u)
        rhs = np.dot(R, rhs)

        # Restrict A
        if i > 0: A = np.dot(R_int, np.dot(A, T_int)) #RAT 
        A_list.append(A)
        print 'line 118'
        new_u, rhs, dx = v_sched(u, A, rhs, 2*dx, num_pre=num_pre, num_post=num_post, num_down=num_down, num_up=num_up, method='GS')
    
    u = new_u.copy()

    # -----------------
    # Interpolate
    # -----------------
    A = A_list.pop()

    for j in xrange(num_up):

        # get T 
        T = get_T(len(rhs))

        # Interpolate u, rhs
        new_u = np.dot(T, u)
        new_rhs = np.dot(T, rhs)

        # Interpolate by getting the last A
        A = A_list.pop()
        print 'line 138'
        u, rhs, dx = v_sched(new_u, A, new_rhs, .5 * dx, num_pre=num_pre, num_post=num_post, num_down=num_down, num_up=num_up, method='GS')
    
    return new_u, rhs, dx

def make_initial_guess(length, wns, N):
    '''
    length: length of guess vector
    wns: a list of wavenumbers
    N: number of points in interval [a, b] except the point a 
       if we have x interior points then we have x+2 total points 
       and so N = x+1
    
    return
    initial guess numpy array

    '''
    numWaveNumbers = len(wns)
    guess = np.zeros(length)
    for i in range(length):
        for k in wns:
            guess[i] += (1./numWaveNumbers) * np.sin(k * i * np.pi / N)
    return guess

def main():
    # Problem setup
    domain = (0.0, 2.0 * np.pi)
    bc = (0., 0.)
    m = 17 # includes boundary points, m-2 interior points
    x = np.linspace(domain[0], domain[1], m)
    dx = x[1] - x[0]
    
    # 1. initial guess of all zeros
    #U0 = np.zeros(m)
    
    # 2. sinusoidal guess
    wavenumbers = [12, 30] # k1, k2
    U0 = make_initial_guess(m, [12, 30], N = m-1)

    #u_true = lambda x: np.sin(x)
    #f = lambda x: - np.sin(x)
    u_true = lambda x: 0.0 * x

    def f(x):
        if type(x) is np.ndarray:
            return np.zeros(len(x))
        if type(x) is float or np.float64:
            return 0.0

    grid = MGC(x, U0, domain, f)
    print grid.A_list[0].shape
    grid.u, rhs, dx = v_sched(grid.u, grid.A_list[0], f(x), dx)
    print grid.u

if __name__ == "__main__":
    main()


