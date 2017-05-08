############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


import numpy as np 
import matplotlib.pyplot as plt
from MultiGridClass import MultiGridClass as MGC
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
	m = 9 # includes boundary points, m-2 interior points
	x = np.linspace(domain[0], domain[1], m)
	
	# 1. initial guess of all zeros
	#U0 = np.zeros(m)
	
	# 2. sinusoidal guess
	wavenumbers = [12, 30] # k1, k2
	U0 = make_initial_guess(m, [12, 30], N = m-1)

	#u_true = lambda x: np.sin(x)
	#f = lambda x: - np.sin(x)
	u_true = lambda x: 0.0
	def f(x):
		if type(x) is np.ndarray:
			return np.zeros(len(x))
		if type(x) is float or np.float64:
			return 0.0

	mg_grid = MGC(x, U0, domain, f)
	mg_grid.solveCGC(u_true = u_true)
	# mg_grid.v_sched(num_pre=2, num_post=2, 
	# 	num_down = 1, num_up =1, u_true=u_true)
	# mg_grid.plot_true()

if __name__ == "__main__":
	main()