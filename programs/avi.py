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

    ## BASE CASE
    if num_down == 0:
        # print 'base case'
        u[1:-1] = np.linalg.solve(A, rhs[1:-1])
        return u, rhs, dx

    
    # -----------------
    # Relax on Au = rhs
    # -----------------

    u = iterative_solver(u, rhs, dx, num_iterations=num_pre, method='GS')
    
    # -----------------
    # Solve defect eq for residue
    # -----------------

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
    # print 'line92'
    e, rhs_e, dx_e = v_sched(np.zeros(len(residue)), A, residue, 2*dx, num_pre=num_pre, num_post=num_post, num_down=num_down-1, num_up=num_up-1, method='GS')
    
    T = get_T(len(e))
    e = np.dot(T, e)

    u = u + e

    # -----------------
    # Relax on Au = rhs
    # -----------------

    u = iterative_solver(u, rhs, dx, num_iterations=num_post, method='GS')

    return u, rhs, dx

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

    u_true = lambda x: -1/36.0 * np.sin(6.0 *x)

    f = lambda x: np.sin(6.0 *x)

    domain = (0.0, 1.0)
    bc = [u_true(i) for i in domain]
    m = 2**6 + 1 # includes boundary points, m-2 interior points
    x = np.linspace(domain[0], domain[1], m)
    dx = x[1] - x[0]

    U0 = np.zeros(len(x))
    U0[0] = bc[0]
    U0[-1] = bc[-1]

    grid = MGC(x, U0.copy(), domain, f, bc=bc)

    jacobi = MGC(x, U0.copy(), domain, f, bc=bc)
    
    grid_u, rhs, dx = v_sched(grid.u, grid.A_list[0], f(x), dx, num_down=5)

    jacobi_u = iterative_solver(jacobi.u, f(x), dx, num_iterations=16)
    
    print grid.u
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    axes.plot(x, grid_u, label='MG')

    axes.plot(x, u_true(x), label='u_true')
    
    axes.plot(x, jacobi_u, label='jacobi')
    axes.legend()
    plt.show()

if __name__ == "__main__":
    main()


