import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def getInterpolation_T(m, level, max_level=4):
	'''
	Parameters/Arguments
	m: the number of discretizations within the interval [a,b]
	   including the boundary points
	level: grid level

	Output/Return
	Returns the matrix interpolating matrix T that interpolates 
	points from a grid level i to a grid level i+1 

	fine to coarse discretization
	
	

	'''
	if level >= max_level:
		print("grid level must be less than the maximum grid level")
		return
	else:
		T = np.zeros((m, m/2 + 2))
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
		print(T)
		return T

def getRestriction_R(m, level, max_level=4):
	''' 
	Parameters/Arguments
	m: the number of discretizations within the interval [a, b]
	   including the boundary points 
	level: the grid level we are currently at

	Output/Return
	Returns the interpolating matrix R that interpolates a re-fining
	of the discretization used at grid level i to a grid level at i-1

	coarse to fine discretization
	'''
	if level <= 1: 
		print("We must be at a level from 2 to the max level")
		return 
	else: 
		R = np.zeros((m, 2*m-2))
		# get corners
		R[0,0] = 1
		R[-1,-1] = 1

		j = 1
		for i in range(R.shape[0]):
			if i >= 1 and i < R.shape[0]-1:
				R[i, j] = .25
				R[i, j+1] = .5
				R[i, j+2] = .25
				j += 1
		print R
		return R

# grid 1: 18 discretizations (16 interior + 2 boundary)
# grid 2: 10 discretizations ( 8 interior + 2 boundary)
# grid 3: 6 discretizations  ( 4 interior + 2 boundary)
# grid 4: 4 discretizations  ( 4 interior + 2 boundary)


def main():
	m = 9
	T = getInterpolation_T(m, level=1)
	R = getRestriction_R(m, level=2)
	x = np.linspace(0, 1, 9)

	print(x, np.dot(x, T))

if __name__ == "__main__":
	main()