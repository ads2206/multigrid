############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################

import seaborn as sns

from MultiGridClass import MultiGridClass
import numpy as np
from matplotlib import pyplot as plt


# Problem setup
m = 2**9+1
domain = (0.0, 2.0*np.pi)
x = np.linspace(domain[0], domain[1], m)
U0 = np.zeros(m)
#U0 = np.sin(np.pi*x)
u_true = lambda x: np.sin(x)
f = lambda x: - np.sin(x)


# m = 17
# x = np.linspace(0,1,17)
# U0 = np.zeros(m)
# domain = (0.0, 1.0)
# f = lambda x: np.exp(x)
# bc = (1, np.exp(1.0))
# U0[0] = 1
# U0[-1] = np.exp(1.0)
# u_true = lambda x: np.exp(x)


fmg_grid = MultiGridClass(x, U0, domain, f, solver='GS')
static_grid = MultiGridClass(x, U0, domain, f, solver='GS')

# plot the initial guess because it is plotting 'self.u' which 
# gets initialized to U0 the guess
fmg_grid.plot(u_true, plot_error=True)

# for v-schedule only 
mg_grid = MultiGridClass(x, U0, domain, f, solver='GS')



for i in range(1):
	mg_grid.v_sched()
	static_grid.iterate(4)
	# for FMG 
	fmg_grid.full_multi_grid()


#static_grid.plot(u_true, plot_error=True)
mg_grid.plot(u_true, plot_error=True)
fmg_grid.plot(u_true, plot_error=True)

plt.show()
