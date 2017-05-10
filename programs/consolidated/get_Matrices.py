############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# get_Matrices.py module
# contains functions for generating one and two
# dimensional restriction and interpolation
# matrices.. T and R are equal to transposes of 
# one another up to a constant, this is used for
# some of the functions.
####

import numpy as np
import scipy.sparse as sparse

def get_Tint(numInt, dim=1):
    '''
    Interpolation for interior dimensions
 
    numInt: number of interior points in [a,b]
    dim: dimension 
    '''
    numInt = int(numInt)

    if dim == 1:
        T_int = np.zeros((2*numInt + 1, numInt))
        for i in range(int(numInt)):
            T_int[2*i:2*i+3, i] = np.array([.5, 1., .5])
        return T_int

    elif dim == 2:
        return 4 * get_Rint(numInt, 2).T

def get_Rint(numInt, dim=1):
    '''
    Restriction for interior dimensions
    
    numInt: number of interior points in [a,b]
    dim: dimension 
    '''

    numInt = int(numInt)

    if dim == 1:
        R = .5*get_Tint(numInt).T
        return R

    elif dim == 2: 
        R = .5*get_Tint(numInt).T
        return np.kron(R, R)

def get_T(n, dim=1):
    '''
    Interpolation for objects with size of entire domain

    numInt: number of interior points in [a,b]
    dim: dimension 
    '''
    if dim == 1:
        T = np.zeros((2*n -1, n ))
         #iterate row by row
        j = k = 0
        for i in range(T.shape[0]):
            if (i % 2) == 0:
                T[i, j] = 1
                j += 1
            else:
                T[i, k] = .5
                T[i, k+1] = .5
                k+=1
        return T

    elif dim == 2:
        T = get_T(n, dim=1)
        R = .5 * T.T
        R_2d = np.kron(R, R)
        return 4 * R_2d.T

def get_R(n, dim=1):
    '''
    Restriction for objects with size of entire domain

    numInt: number of interior points in [a,b]
    dim: dimension 
    '''
    T = get_T(n, dim=1)
    R = .5*T.T
    R[0, 0:2] = [1, 0]
    R[-1, -2:] = [0, 1]

    if dim == 1:
        return R

    elif dim == 2:
        # restriction matrix R2D in 2-dim; source: Strang
        return np.kron(R, R)
