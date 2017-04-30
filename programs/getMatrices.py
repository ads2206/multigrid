import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def get_T(n, dim=1):
	'''
	Parameters/Arguments
	n: the number of discretizations within the interval [a,b]
	   including the boundary points

	Output/Return
	Returns the matrix interpolating matrix T that interpolates 
	points from a grid level i to a grid level i+1 

	coarse to fine interpolation
	9 by 5
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
		return T

def get_R(n, dim=1):
	''' 
	Parameters/Arguments
	n: the number of discretizations within the interval [a, b]
	   including the boundary points 
	level: the grid level we are currently at

	Output/Return
	Returns the interpolating matrix R that interpolates a re-fining
	of the discretization used at grid level i to a grid level at i-1. 
	The coarse grid vector (the output), takes its value directly from 
	the corresponding fine grid point.

	fine to coarse restriction
	'''
	if dim == 1;
		row = range(n)
		col = [2*i for i in row]

		R = np.zeros((n, 2*n-1))
		R[row, col] = 1
		return R
	elif dim ==2:

		return R
		# potential 

# grid 1: 17 discretizations (15 interior + 2 boundary)
# grid 2: 9 discretizations  ( 7 interior + 2 boundary)
# grid 3: 5 discretizations  ( 3 interior + 2 boundary)
# grid 4: 3 discretizations  ( 1 interior + 2 boundary)
def main():
	n = 5
	T = get_T(n)
	R = get_R(n)
	x = np.linspace(0, 1, 2*n-1)
	print("input vector: ", x)
	print("Restriction.    R dot x: ", np.dot(R, x))
	x = np.linspace(0, 1, n)
	print("input vector: ", x)
	print("Interpolation.  T dot x: ", np.dot(T, x))

if __name__ == "__main__":
	main()