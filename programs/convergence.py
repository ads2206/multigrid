
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

m_list = [2**i + 1 for i in range(2, 10)]

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

    fmg_grid = MultiGrid1d(x, U0, domain, f, solver='GS')


    #-----------------------
    # Solving
    #-----------------------

    for i in range(1):
        fmg_grid.fmg() 
    error.append(fmg_grid.get_error(u_true))
    
#-----------------------
# Plotting
#-----------------------
    
fig = plt.figure()
axes = fig.add_subplot(1,1,1)
plt.semilogy(m_list, error, label='FMG')
plt.legend()
plt.xlabel('m value')
plt.ylabel('error')
axes.set_title('Full Multi Grid Cycle Error Convergence')

txt = '''This shows the error in an approximation made by one
cycle of Full Multi Grid Cycle at 8 different descritizations, 
moving from $ m = 5$ to $m = 1025$ on the domain $[0, 2 \pi]$.'''


# Show or Save?
# plt.show()
plt.savefig('../plots/fmg_error.pdf')


