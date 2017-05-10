############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################


####
# One dimensional example solving Au = 0
# Shows the impact of the coarse grid correction scheme
####

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from MultiGrid import MultiGrid1d, MultiGrid2d

def make_initial_guess(length, wns, N):
    '''
    length: length of guess vector
    wns: a list of wavenumbers
    N: number of points in interval [a, b] except the point a 
       if we have x interior points then we have x+2 total points 
       and so N = x+1
    
    return
    initial guess numpy array

    '''
    numWaveNumbers = len(wns)
    guess = np.zeros(length)
    for i in range(length):
        for k in wns:
            guess[i] += (1./numWaveNumbers) * np.sin(k * i * np.pi / N)
    return guess

#-----------------------
# Problem setup
#-----------------------

m = 2**9+1
domain = (0.0, 2.0*np.pi)
x = np.linspace(domain[0], domain[1], m)
u_true = lambda x: 0 * x
f = lambda x: 0 * x

# Sinusoidal guess
wavenumbers = [12, 30] # k1, k2
U0 = make_initial_guess(m, [12, 30], N = m-1)

#-----------------------
# Initializaion
#-----------------------

v_grid = MultiGrid1d(x, U0, domain, f, solver='GS')


#-----------------------
# Solving
#-----------------------

# store values for error comparison:
iterations = [0]
solutions = []

for i in range(6):
    v_grid.v_sched() 
    solutions.append((v_grid.x, v_grid.u))

#-----------------------
# Plotting
#-----------------------
fig = plt.figure()
j = 1
for i in xrange(len(solutions)):
    if i%3 == 0:
        axes = fig.add_subplot(2,1,j)
        j += 1
    x = solutions[i][0]
    u = solutions[i][1]
    axes.plot(x, u, label=str(i))
    if i%3 == 2:
        plt.legend()
        plt.xlabel('effective iterations')
        plt.ylabel('error')

axes.set_title('An oscillitory initial guess converging to zero')

# Show or Save?
# plt.show()
plt.savefig('../plots/zero_1d.pdf')


