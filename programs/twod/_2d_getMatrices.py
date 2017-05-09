import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def get_Tint(numInt, dim=1):
	'''
	numInt: number of interior points in [a,b]
	dim: dimension we are in

	need to implement 2d!
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
	numInt: number of interior points in [a,b]
	dim: dimension we are in

	NEED To implement 2-dimension 
	'''

	if dim == 1:
		# talk to avi about this A_list and rseidual issue
		#return .5*get_Tint( (numInt-1) /2 ).T
		R = .5*get_Tint(numInt).T
		return R
	elif dim == 2: 
		R = .5*get_Tint(numInt).T
		return np.kron(R, R)

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
		T = get_T(n, dim=1)
		R = .5*T.T
		R_2d = np.kron(R, R)
		# R_2d = .25 * T_2d^tranpose; source: strang
		return 4.* R_2d.T

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
	T = get_T(n, dim=1)
	R = .5*T.T
	R[0, 0:2] = [1, 0]
	R[-1, -2:] = [0, 1]

	if dim == 1:

		# *** injection method below ***
		# row = range(n)
		# col = [2*i for i in row]

		# R = np.zeros((n, 2*n-1))
		# R[row, col] = 1
		return R
	elif dim == 2:

		# 3 by 7 matrix R in 1d becomes a 9 by 49 
		# restriction matrix R2D in 2-dim; source: Strang
		return np.kron(R, R)

# grid 1: 17 discretizations (15 interior + 2 boundary)
# grid 2: 9 discretizations  ( 7 interior + 2 boundary)
# grid 3: 5 discretizations  ( 3 interior + 2 boundary)
# grid 4: 3 discretizations  ( 1 interior + 2 boundary)
def main():
	np.set_printoptions(threshold=np.nan, linewidth=10000)
	m = 7
	e = np.ones(m)
	A = sparse.spdiags([e, -2*e, e], [-1,0,1], m, m).toarray()
	print A
	print ''
	print ''
	temp = np.dot(A, get_Tint(3))
	print np.dot(get_Rint(3), temp)


	print ''
	print ''
	print 'Two D'
	m = 15
	e = np.ones(m)
	T = sparse.spdiags([e, -4 * e, e], [-1, 0, 1], m, m)
	S = sparse.spdiags([e, e], [-1, 1], m, m)
	I = sparse.eye(m)
	A = sparse.kron(I, T) + sparse.kron(S, I).toarray()

	print A
	temp = np.dot(A, get_Tint(7, 2))
	A = np.dot(get_Rint(7, 2), temp) #RAT

	print 4*A

	temp = np.dot(A, get_Tint(3, 2))
	print 16*np.dot(get_Rint(3, 2), temp)
if __name__ == "__main__":
	main()


		# 1-d Testing
	# n = 5
	# T = get_T(n, dim=1)
	# R = get_R(n, dim=1)
	
	# x = np.linspace(0, 1, n)
	# print("input vector: ", x)
	# print("Restriction.    R dot x: ", np.dot(R, x))
	
	# x = np.linspace(0, 1, n)
	# print("input vector: ", x)
	# print("Interpolation.  T dot x: ", np.dot(T, x))

	# #2-d Testing 
	# x = 2.*np.ones((2*n-1)**2)
	# T_2d = get_T(n, dim=2)
	# R_2d = get_R(n, dim=2)
	# out =np.dot(R_2d, x).reshape(n, n)
	# print(out[0,:], out[-1,:], out[:,0], out[:, -1])
	# print(out[1:-1][:, 1:-1])



	# print 'One D'
	# print ''
	# big = np.ones(9)
	# big[0] = 0
	# big[-1] = 0
	# print get_R(5)
	# small = np.dot(get_R(5), big)
	# new_big = np.dot(get_T(5), small)

	# print big
	# print ''
	# print small
	# print new_big
	# print ''
	# print''

	# print 'Two D'
	# print ''
	# numInt = 7
	# big = np.zeros((9,9))
	# big[1:-1, 1:-1] = np.ones((7,7))
	# print get_R(5, dim=2)
	# small = np.dot(get_R(5, dim=2), big.reshape(81))
	# new_big = np.dot(get_T(5, dim=2), small)

	# print big
	# print ''
	# print small.reshape((5,5))
	# print new_big.reshape((9,9))
	# print ''
	# print''	

	# print 'One D'
	# print ''
	# big = np.ones(9)
	# print get_R(5)
	# small = np.dot(get_R(5), big)
	# new_big = np.dot(get_T(5), small)

	# print big
	# print ''
	# print small
	# print new_big
	# print ''
	# print''

	# print 'Two D'
	# print ''
	# numInt = 7
	# big = np.ones((9,9))
	# print get_R(5, dim=2)
	# small = np.dot(get_R(5, dim=2), big.reshape(81))
	# new_big = np.dot(get_T(5, dim=2), small)

	# print big
	# print ''
	# print small.reshape((5,5))
	# print new_big.reshape((9,9))
	# print ''
	# print''

	# big = np.ones((7,7))
	# small = np.dot(get_Rint(3, dim=2), big.reshape(49))
	# new_big = np.dot(get_Tint(3, dim=2), small)

	# print big
	# print small.reshape((3,3))
	# print new_big.reshape((7,7))

	# print(16*get_T(5, dim=2))
	# print(16*get_R(5, dim=2))
