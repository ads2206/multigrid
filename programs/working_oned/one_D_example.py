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
m = 17
domain = (0.0, 2.0 * np.pi)
x = np.linspace(domain[0], domain[1], m)
U0 = np.zeros(m)
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


mg_grid = MultiGridClass(x, U0, domain, f, solver='GS')
static_grid = MultiGridClass(x, U0, domain, f, solver='GS')
mg_grid.plot(u_true, plot_error=True)

mg_grid.v_sched()
static_grid.iterate(4)

static_grid.plot(u_true, plot_error=True)
mg_grid.plot(u_true, plot_error=True)

plt.show()
