############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################

import seaborn as sns

from mg2d import MultiGridClass
import numpy as np
from matplotlib import pyplot as plt

#-----------------------
# Problem setup
#-----------------------

# f = lambda x, y: 1.25 * np.exp(x + y / 2.0)
# u_true = lambda x, y: np.exp(x + y / 2.0)

# m = 9
# domain = (0.0, 1.0, 0.0, 1.0)
# x = np.linspace(0.0, 1.0, m)
# y = np.linspace(0.0, 1.0, m)
# U0 = np.zeros((m, m))


# # Boundary Functions
# alpha_x = lambda y: np.exp(y/2.0)
# beta_x = lambda y: np.exp(1.0 + y/2.0)
# alpha_y = lambda x: np.exp(x)
# beta_y = lambda x: np.exp(x + 0.5)
# bc = (alpha_x, beta_x, alpha_y, beta_y)

f = lambda x, y: -.5 * np.sin(x)*np.sin(y)
u_true = lambda x, y: 1.0 / 4.0 * np.sin(x)*np.sin(y)

m = 33
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
#-----------------------
# Test
#-----------------------

mg_grid = MultiGridClass(x, y, U0, domain, f, bc, solver='GS')
sor_grid = MultiGridClass(x, y, U0, domain, f, bc, solver='SOR')
# gs_grid = MultiGridClass(x, y, U0, domain, f, bc, solver='GS')

for i in range(1):
    # gs_grid.iterate_2d(4)
    sor_grid.fmg_2d()
    mg_grid.v_sched_2d()

mg_grid.plot_2d(u_true, plot_error=False)
sor_grid.plot_2d(u_true, plot_error=False)
# gs_grid.plot_2d(u_true, plot_error=False)

plt.show()


# mg_grid = MultiGridClass(x, U0, domain, f, solver='GS')
# sor_grid = MultiGridClass(x, U0, domain, f, solver='SOR')
# static_grid = MultiGridClass(x, U0, domain, f, solver='GS')

# # mg_grid.plot(u_true, plot_error=True)

# iterations = [0]
# error = [(mg_grid.get_error(u_true), static_grid.get_error(u_true), sor_grid.get_error(u_true))]
# # error = [(static_grid.get_error(u_true), sor_grid.get_error(u_true))]

# for i in range(125):
#     mg_grid.v_sched()
#     static_grid.iterate(4)
#     sor_grid.iterate(4)
#     error.append((mg_grid.get_error(u_true), static_grid.get_error(u_true), sor_grid.get_error(u_true)))
#     iterations.append(static_grid.iter_count)
# # static_grid.plot(u_true, plot_error=True)
# # mg_grid.plot(u_true, plot_error=True)

# # print mg_grid.iter_count
# # print static_grid.iter_count
# fig = plt.figure()
# axes = fig.add_subplot(1,1,1)
# plt.semilogy(iterations, [i[0] for i in error], label='MG')
# plt.semilogy(iterations, [i[1] for i in error], label='GS')
# plt.semilogy(iterations, [i[2] for i in error], label='SOR')
# plt.legend()
# plt.xlabel('effective iterations')
# plt.ylabel('error')

# mg_grid.plot(u_true, True)

# plt.show()

