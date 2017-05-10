############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# One dimensional example solving u'' = f(x)
####

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from MultiGrid import MultiGrid1d, MultiGrid2d

#-----------------------
# Problem setup
#-----------------------

m = 2**9+1
domain = (0.0, 2.0*np.pi)
x = np.linspace(domain[0], domain[1], m)
U0 = np.zeros(m)
u_true = lambda x: np.sin(x)
f = lambda x: - np.sin(x)

#-----------------------
# Initializaion
#-----------------------

# Initialize grids for comparison: [gs_grid, sor_grid, v_grid, fmg_grid]
gs_grid = MultiGrid1d(x, U0, domain, f, solver='GS')
sor_grid = MultiGrid1d(x, U0, domain, f, solver='SOR')
v_grid = MultiGrid1d(x, U0, domain, f, solver='GS')
fmg_grid = MultiGrid1d(x, U0, domain, f, solver='GS')

# plot initial guess (since they are all the same we can choose any one)
gs_grid.plot(u_true, plot_error=False)

#-----------------------
# Solving
#-----------------------

# store values for error comparison:
iterations = [0]
error = [(gs_grid.get_error(u_true), sor_grid.get_error(u_true), 
            v_grid.get_error(u_true), fmg_grid.get_error(u_true))]

for i in range(10):
    # Call iterate(4) so that the iteration counts match the other two
    gs_grid.iterate(4) 
    sor_grid.iterate(4)
    v_grid.v_sched() 
    fmg_grid.fmg() 

    error.append((gs_grid.get_error(u_true), sor_grid.get_error(u_true), 
            v_grid.get_error(u_true), fmg_grid.get_error(u_true)))
    iterations.append(gs_grid.iter_count)

#-----------------------
# Plotting
#-----------------------
gs_grid.plot(u_true, plot_error=True)
plt.savefig('../plots/gs_sol_1d.pdf')

sor_grid.plot(u_true, plot_error=True)
plt.savefig('../plots/sor_sol_1d.pdf')

v_grid.plot(u_true, plot_error=True)
plt.savefig('../plots/v_sol_1d.pdf')

fmg_grid.plot(u_true, plot_error=True)
plt.savefig('../plots/fmg_sol_1d.pdf')


fig = plt.figure()
axes = fig.add_subplot(1,1,1)
plt.semilogy(iterations, [i[0] for i in error], label='GS')
plt.semilogy(iterations, [i[1] for i in error], label='SOR')
plt.semilogy(iterations, [i[2] for i in error], label='V-Grid')
plt.semilogy(iterations, [i[3] for i in error], label='FMG')
plt.legend()
plt.xlabel('effective iterations')
plt.ylabel('error')


# Show or Save?
# plt.show()
plt.savefig('../plots/error_1d.pdf')


