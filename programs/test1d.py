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

def main():

	# Problem setup
	domain = (0.0, 2.0 * np.pi)
	bc = (0., 0.)
	m = 9
	x = np.linspace(domain[0], domain[1], m)
	U0 = np.zeros(m)
	u_true = lambda x: np.sin(x)
	f = lambda x: - np.sin(x)

	mg_grid = MGC(x, U0, domain, f)
	mg_grid.v_sched(u_true=u_true)


	mg_grid.plot()

if __name__ == "__main__":
	main()