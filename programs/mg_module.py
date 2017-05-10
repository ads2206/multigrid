############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# Multi Grid Module 
# iterative_solver, iterative_solver_2d, 
# v_sched, v_sched_2d, fmg, fmg_2d
####

import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.interpolate import interp1d, interp2d
from get_Matrices import get_T, get_R, get_Tint, get_Rint

def iterative_solver(u, rhs, dx, num_iterations=1, method='GS'):
        ''' Approximiating u in 1D system Au = rhs using an iterative method
        *input*
        u:  size total boundary, including endpoints, input is initial guess
        rhs: size total boundary, including endpoints, input should be unchanged

        *output*
        u: size unchanges, but closer approximation to the solution'''
        
        # Set omega for SOR
        omega = 2.0 - 2 * np.pi * dx

        m  = len(rhs)
        for j in range(num_iterations):
            for i in range(1, m - 1):

                # interior jacobi iteration
                if method=='GS':
                    u[i] = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])

                elif method=='SOR':
                    u_gs = 0.5 * (u[i-1] + u[i+1] - dx**2 * rhs[i])    
                    u[i] = u[i] + omega * (u_gs - u[i])
        return u

def iterative_solver_2d(u, rhs, dx, num_iterations=1, method='GS'):
        ''' Approximiating u in 2D system Au = rhs using an iterative method
        *input*
        u:  size total boundary, including endpoints, input is initial guess
        rhs: size total boundary, including endpoints, input should be unchanged

        *output*
        u: size unchanges, but closer approximation to the solution'''
        # Set omega for SOR
        
        omega2 = 2.0 - 2 * np.pi * dx

        m  = len(rhs)
        for k in range(num_iterations):
            for i in range(1, m - 1):
                for j in range(1, m - 1):

                    if method=='GS':
                        u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j-1] + u[i, j+1] - dx**2 * rhs[i, j])

                    elif method=='SOR':
                        u_gs = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j-1] + u[i, j+1] - dx**2 * rhs[i, j])
                        u[i, j] = u[i, j] + omega2 * (u_gs - u[i, j])
        return u

def v_sched(u, A, rhs, dx, num_pre=2, num_post=2, level=2, method='GS'):
    ''' Approximiating u in 2D system Au = rhs using multigrid technique of
    one multi grid cycle.

    *input*
    u:  size total boundary, including endpoints, input is initial guess
    rhs: size total boundary, including endpoints, input should be unchanged
    A: dimenssions of the interior of the problem

    *output*
    u: size unchanged, but closer approximation to the solution'''

    ## BASE CASE
    if level == 0:
        u[1:-1] = np.linalg.solve(A, rhs[1:-1])
        return u
    
    # -----------------
    # Relax on Au = rhs (num_pre)
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
    e = v_sched(np.zeros(len(residue)), A, residue, 2.0 * dx, num_pre=num_pre, num_post=num_post, level=level-1, method='GS')
    
    T = get_T(len(e))
    e = np.dot(T, e)

    u = u + e

    # -----------------
    # Relax on Au = rhs (num_post)
    # -----------------
    u = iterative_solver(u, rhs, dx, num_iterations=num_post, method='GS')

    return u

def v_sched_2d(u, A, rhs, dx, num_pre=2, num_post=2, level=2, method='GS'):
    ''' Approximiating u in 2D system Au = rhs using multigrid technique of
    a single multi grid cycle.
    *input*
    u:  size total boundary, shape is square, input is initial guess
    rhs: size total boundary, shape is square, input should be unchanged
    A: dimenssions of the interior of the problem
    dx: delta x in the current descretization

    *output*
    u: size unchanged, but closer approximation to the solution'''

    ## BASE CASE
    if level == 1:
        rhs_vec = rhs[1:-1, 1:-1].reshape((len(rhs)-2)**2)
        u_vec = np.linalg.solve(A, rhs_vec)
        u[1:-1, 1:-1] = u_vec.reshape(len(rhs)-2, len(rhs)-2)
        return u
    
    # -----------------
    # Relax on Au = rhs (num_pre)
    # -----------------
    u = iterative_solver_2d(u, rhs, dx, num_iterations=num_pre, method=method)
    
    # -----------------
    # Solve defect eq for residue
    # -----------------
    residue = np.zeros((len(rhs), len(rhs)))

    rhs_vec = rhs[1:-1, 1:-1].reshape((len(rhs)-2)**2)
    u_vec = u[1:-1, 1:-1].reshape((len(u)-2)**2)
    residue_vec = rhs_vec - np.dot(A, u_vec)
    
    residue[1:-1, 1:-1] = residue_vec.reshape((len(residue)-2, len(residue)-2))

    R = get_R(len(residue)/2 +1, 2)
    residue_vec = np.dot(R, residue.reshape(len(residue)**2))
    residue = residue_vec.reshape((len(residue)/2+1, len(residue)/2+1))

    # Get size of R_int, T_int
    m = len(A) # get length of one side of A
    m = np.sqrt(m) # get square root, which is the length of the interior
    m = (m - 1) / 2 # get length of new iterior after restriction

    R_int = get_Rint(m, 2)
    T_int = get_Tint(m, 2)
    temp = np.dot(A, T_int)
    A = np.dot(R_int, temp) #RAT     

    # -----------------
    # Get error and use to improve u
    # -----------------
    e = v_sched_2d(np.zeros((len(residue), len(residue))), A, residue, 2.0 * dx, num_pre=num_pre, num_post=num_post, level=level-1, method=method)
    
    T = get_T(len(e), 2)

    e = np.dot(T, e.reshape(len(e)**2))
    e = e.reshape((len(u), len(u)))

    u = u + e

    # -----------------
    # Relax on Au = rhs (num_post)
    # -----------------
    u = iterative_solver_2d(u, rhs, dx, num_iterations=num_post, method=method)

    return u

def fmg(u, A, rhs, dx, level, domain, num_pre=2, num_post=2, method='GS'):
    ''' Approximiating u in 1D system Au = rhs using multigrid technique of
    Full multi grid cycle, calling v_sched() defined above at each level.

    *input*
    u:  size total boundary, shape is square, input is initial guess
    rhs: size total boundary, shape is square, input should be unchanged
    A: dimenssions of the interior of the problem
    dx: delta x in the current descritization

    *output*
    u: size unchanged, but closer approximation to the solution'''
    
    # base case
    if level == 2:
        u[1:-1] = np.linalg.solve(A, rhs[1:-1] )
        return u 
    
    u_temp = u.copy()
    
    R_int = get_Rint(len(A[::2]) - 1)
    T_int = get_Tint(len(A[::2]) - 1)

    R = get_R(len(rhs) / 2 + 1 )

    # smaller A 
    A_small = np.dot(R_int, np.dot(A, T_int)) #RAT 

    rhs_small = R.dot(rhs)
    u_small = R.dot(u_temp)


    # recursive call, terminates by changing level variable
    u_temp = fmg(u_small, A_small, rhs_small, 2*dx, level-1, domain,
                    num_pre=num_pre, num_post=num_post, method=method)

    T = get_T(len(u_temp))


    x = np.arange(domain[0], domain[1]+2.0*dx, 2.0*dx)
    
    u_interp_func = interp1d(x, u_temp, kind=3)

    u_interp = u_interp_func( np.arange(domain[0], domain[1]+dx, dx) )

    return v_sched(u_interp, A, rhs, dx, level=level)

def fmg_2d(u, A, rhs, dx, level, domain, num_pre=2, num_post=2, method='GS'):
    ''' Approximiating u in 2D system Au = rhs using multigrid technique of
    Full multi grid cycle, calling v_sched_2d() defined above at each level.

    *input*
    u:  size total boundary, shape is square, input is initial guess
    rhs: size total boundary, shape is square, input should be unchanged
    A: dimenssions of the interior of the problem
    dx: delta x in the current descritization

    *output*
    u: size unchanged, but closer approximation to the solution'''    

    # base case
    if level == 2:
        rhs_vec = rhs[1:-1, 1:-1].reshape((len(rhs)-2)**2)
        u_vec = np.linalg.solve(A, rhs_vec)
        u[1:-1, 1:-1] = u_vec.reshape(len(rhs)-2, len(rhs)-2)
        return u
    
    
    # Get size of R_int, T_int
    m = len(A) # get length of one side of A
    m = np.sqrt(m) # get square root, which is the length of the interior
    m = (m - 1) / 2 # get length of new iterior after restriction
    
    R_int = get_Rint(m, dim=2)
    T_int = get_Tint(m, dim=2)

    R = get_R(len(rhs) / 2 + 1, dim=2 )

    # smaller A 
    A_small = np.dot(R_int, np.dot(A, T_int)) #RAT 

    rhs_vec = rhs.reshape(len(rhs)**2)
    rhs_small = R.dot(rhs_vec).reshape((len(rhs)/2+1, len(rhs)/2+1))

    u_vec = u.reshape(len(u)**2)
    u_small = R.dot(u_vec).reshape((len(u)/2+1, len(u)/2+1))


    # recursive call, terminates by changing level variable
    u_temp = fmg_2d(u_small, A_small, rhs_small, 2*dx, level-1, domain,
                    num_pre=num_pre, num_post=num_post, method=method)



    x = np.arange(domain[0], domain[1]+2.0*dx, 2.0*dx)
    y = np.arange(domain[2], domain[3]+2.0*dx, 2.0*dx)
    X, Y = np.meshgrid(x, y)
    
    
    u_interp_func = interp2d(x, y, u_temp, kind='cubic')

    x = np.arange(domain[0], domain[1]+dx, dx)
    y = np.arange(domain[2], domain[3]+dx, dx)
    X, Y = np.meshgrid(x, y)
    u_interp = u_interp_func(x, y)

    return v_sched_2d(u_interp, A, rhs, dx, level=level)
