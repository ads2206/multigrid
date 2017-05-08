############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################

from MultiGrid2D import MultiGrid2D
import numpy as np
from matplotlib import pyplot as plt


# Problem setup
f = lambda x, y: 1.25 * np.exp(x + y / 2.0)
u_true = lambda x, y: np.exp(x + y / 2.0)

m = 17
domain = (0.0, 1.0, 0.0, 1.0)
x = np.linspace(0.0, 1.0, m)
y = np.linspace(0.0, 1.0, m)
U0 = np.zeros((m, m))
# Boundary Functions
alpha_x = lambda y: np.exp(y/2.0)
beta_x = lambda y: np.exp(1.0 + y/2.0)
alpha_y = lambda x: np.exp(x)
beta_y = lambda x: np.exp(x + 0.5)
bc = (alpha_x, beta_x, alpha_y, beta_y)


mg_grid = MultiGrid2D(x, y, U0, domain, f, bc, u_true)

# X, Y = np.meshgrid(x, y)
static_grid = MultiGrid2D(x, y, U0, domain, f, bc, u_true)
static_grid.iterative_solver(24)

# mg_grid.plot(u_true)

mg_grid.v_sched(u_true=u_true)
mg_grid.v_sched(u_true=u_true)
mg_grid.v_sched(u_true=u_true)

# mg_grid.iterative_solver(1000)
# # print mg_grid.get_error(u_true)
# mg_grid.test()
# mg_grid.plot(u_true)

# mg_grid.restrict()
# mg_grid.plot(u_true)

# mg_grid.test()
# mg_grid.interpolate()
mg_grid.plot(u_true, static_grid.u)

# print mg_grid.get_error(u_true)


# mg_grid.plot(u_true)
# static_grid.plot()

