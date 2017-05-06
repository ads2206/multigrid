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
m = 2**8+1
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
sor_grid = MultiGridClass(x, U0, domain, f, solver='SOR')
static_grid = MultiGridClass(x, U0, domain, f, solver='GS')

# mg_grid.plot(u_true, plot_error=True)

iterations = [0]
error = [(mg_grid.get_error(u_true), static_grid.get_error(u_true), sor_grid.get_error(u_true))]
# error = [(static_grid.get_error(u_true), sor_grid.get_error(u_true))]

for i in range(125):
    mg_grid.v_sched()
    static_grid.iterate(4)
    sor_grid.iterate(4)
    error.append((mg_grid.get_error(u_true), static_grid.get_error(u_true), sor_grid.get_error(u_true)))
    iterations.append(static_grid.iter_count)
# static_grid.plot(u_true, plot_error=True)
# mg_grid.plot(u_true, plot_error=True)

# print mg_grid.iter_count
# print static_grid.iter_count
fig = plt.figure()
axes = fig.add_subplot(1,1,1)
plt.semilogy(iterations, [i[0] for i in error], label='MG')
plt.semilogy(iterations, [i[1] for i in error], label='GS')
plt.semilogy(iterations, [i[2] for i in error], label='SOR')
plt.legend()
plt.xlabel('effective iterations')
plt.ylabel('error')

plt.show()
