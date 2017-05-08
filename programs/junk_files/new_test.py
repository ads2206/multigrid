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
f = lambda x, y: -2 * np.sin(x)*np.sin(y)
u_true = lambda x, y: np.sin(x)*np.sin(y)

m = 17
domain = (0.0, 2 * np.pi, 0.0, 2 * np.pi)
x = np.linspace(0.0, 2 * np.pi, m)
y = np.linspace(0.0, 2 * np.pi, m)
U0 = np.zeros((m, m))
# Boundary Functions
alpha_x = lambda y: 0 * y
beta_x = lambda y: 0 * y
alpha_y = lambda x: 0 * x
beta_y = lambda x: 0 * x
bc = (alpha_x, beta_x, alpha_y, beta_y)


mg_grid = MultiGrid2D(x, y, U0, domain, f, bc, u_true, 'multi')
# static_grid = MultiGrid2D(x, y, U0, domain, f, bc, u_true, 'bad')


# static_grid.iterative_solver(150)
mg_grid.v_sched()


# print mg_grid.u - static_grid.u

mg_grid.plot(u_true)
# static_grid.plot(u_true)

# fig = plt.figure()
# fig.set_figwidth(fig.get_figwidth())
# axes = fig.add_subplot(1, 1, 1)
# axes.semilogy(mg_grid.error[::3], label='MultiGrid')
# axes.semilogy(static_grid.error[::3], label='regular Jacobi')
# axes.legend()
# plt.show()