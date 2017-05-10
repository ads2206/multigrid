
############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# Error analysis to show convergence we delta x
####

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from MultiGrid import MultiGrid1d, MultiGrid2d

#-----------------------
# Problem setup
#-----------------------
m_list = [2**i + 1 for i in range(2, 5)]

error = []


for m in m_list:
    domain = (0.0, 2.0*np.pi)
    x = np.linspace(domain[0], domain[1], m)
    U0 = np.zeros(m)
    u_true = lambda x: np.sin(x)
    f = lambda x: - np.sin(x)

    #-----------------------
    # Initializaion
    #-----------------------
    v_grid = MultiGrid1d(x, U0, domain, f, solver='GS')

    #-----------------------
    # Solving
    #-----------------------
    for i in range(m**2):
        v_grid.v_sched() 
    error.append(v_grid.get_error(u_true))

#-----------------------
# Plotting
#-----------------------

fig = plt.figure()
axes = fig.add_subplot(1,1,1)
plt.semilogy(m_list, error, label='V-Grid')
plt.legend()
plt.xlabel('m value')
plt.ylabel('error')
axes.set_title('V-Cycle Error Convergence')

# Show or Save?
# plt.show()
plt.savefig('../plots/v_error.pdf')


