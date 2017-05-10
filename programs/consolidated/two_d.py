############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# Two dimensional example solving u_xx + u_yy = f(x, y)
####

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from MultiGrid import MultiGrid1d, MultiGrid2d

#-----------------------
# Problem setup
#-----------------------

m = 2**5+1
domain = (0.0, 2 * np.pi, 0.0, 2 * np.pi)
x = np.linspace(0.0, domain[1], m)
y = np.linspace(0.0, domain[3], m)
U0 = np.zeros((m, m))
f = lambda x, y: -.5 * np.sin(x)*np.sin(y)
u_true = lambda x, y: 1.0 / 4.0 * np.sin(x)*np.sin(y)


#-----------------------
# Initializaion
#-----------------------

# Initialize grids for comparison: [gs_grid, sor_grid, v_grid, fmg_grid]
gs_grid = MultiGrid2d(x, y, U0, domain, f, solver='GS')
sor_grid = MultiGrid2d(x, y, U0, domain, f, solver='SOR')
v_grid = MultiGrid2d(x, y, U0, domain, f, solver='GS')
fmg_grid = MultiGrid2d(x, y, U0, domain, f, solver='GS')

# plot initial guess (since they are all the same we can choose any one)
gs_grid.plot(u_true)


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
gs_grid.plot(u_true)
plt.savefig('../../plots/gs_sol_2d.pdf')

sor_grid.plot(u_true)
plt.savefig('../../plots/sor_sol_2d.pdf')

v_grid.plot(u_true)
plt.savefig('../../plots/v_sol_2d.pdf')

fmg_grid.plot(u_true)
plt.savefig('../../plots/fmg_sol_2d.pdf')


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
plt.savefig('../../plots/error_2d.pdf')



