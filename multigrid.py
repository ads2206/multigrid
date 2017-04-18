############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################

import numpy
from matplotlib import pyplot as plt

# Problem setup
m = 15
a = 0.0
b = 1.0
u_a = 0.0
u_b = 3.0
f = lambda x: numpy.exp(x)

def interp(x_fine, x_coarse, f):
    raise NotImplemented

########
# Adapted from 06_iterative.ipynb by Prof. Kyle Mandli
########

def solve_jacobi(m, a, b, u_a, u_b, f, U_0=None):


    # Descretization
    x_bc = numpy.linspace(a, b, m + 2)
    x = x_bc[1:-1]
    delta_x = (b - a) / (m + 1)

    # Expected iterations needed
    # iterations_J = int(2.0 * numpy.log(delta_x) / numpy.log(1.0 - 0.5 * numpy.pi**2 * delta_x**2))
    iterations_J = 4

    # Solve system
    # Initial guess for iterations

    if U_0 != None:
        U_new = U_0
        if m < len(U_0) - 2:
            U_new = U_new[::2]
        if m > len(U_0) - 2:
            U_new = interp(x_bc[::2], x_bc, U_0)
    else:
        U_new = numpy.zeros(m + 2)
        U_new[0] = u_a
        U_new[-1] = u_b

    # convergence_J = numpy.zeros(iterations_J)
    step_size_J = numpy.zeros(iterations_J)
    for k in xrange(iterations_J):
        U = U_new.copy()
        for i in xrange(1, m + 1):
            U_new[i] = 0.5 * (U[i+1] + U[i-1]) - f(x_bc[i]) * delta_x**2 / 2.0

        step_size_J[k] = numpy.linalg.norm(U - U_new, ord=2)
        # convergence_J[k] = numpy.linalg.norm(u_true(x_bc) - U_new, ord=2)

    return U, x_bc


def main():
    u_true = lambda x: (4.0 - numpy.exp(1.0)) * x - 1.0 + numpy.exp(x)
    # U_0 = u_true(numpy.linspace(a, b, m+2))
    U1, x_bc1 = solve_jacobi(m, a, b, u_a, u_b, f) 

    U2, x_bc2 = solve_jacobi(m/2, a, b, u_a, u_b, f, U1)  
    
    U3, x_bc3 = solve_jacobi(m/4, a, b, u_a, u_b, f, U2)  

    U4, x_bc4 = solve_jacobi(m/8, a, b, u_a, u_b, f, U3)  

    U5, x_bc5 = solve_jacobi(m/4, a, b, u_a, u_b, f, U4)  



    # Plot result
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(x_bc1, U1, '-or', label="m=15")
    axes.plot(x_bc2, U2, '-ob', label="m=7")
    axes.plot(x_bc3, U3, '-oy', label="m=3")
    axes.plot(x_bc4, U4, '-og', label="m=1") 
    axes.plot(x_bc1, u_true(x_bc1), 'k', label="True")
    axes.set_title("Solution to $u_{xx} = e^x$")
    axes.set_xlabel("x")
    axes.set_ylabel("u(x)")
    axes.legend(loc=2)

    # fig = plt.figure()
    # fig.set_figwidth(fig.get_figwidth() * 2)
    # axes = fig.add_subplot(1, 2, 1)
    # axes.semilogy(range(iterations_J), step_size_J, 'o')
    # axes.semilogy(range(iterations_J), numpy.ones(iterations_J) * delta_x**2, 'r--')
    # axes.set_title("Subsequent Step Size - J")
    # axes.set_xlabel("Iteration")
    # axes.set_ylabel("$||U^{(k)} - U^{(k-1)}||_2$")
    # axes = fig.add_subplot(1, 2, 2)
    # axes.semilogy(range(iterations_J), convergence_J, 'o')
    # axes.semilogy(range(iterations_J), numpy.ones(iterations_J) * delta_x**2, 'r--')
    # axes.set_title("Convergence to True Solution - J")
    # axes.set_xlabel("Iteration")
    # axes.set_ylabel("$||u(x) - U^{(k-1)}||_2$")

    plt.show()

if __name__ == "__main__":
    main()