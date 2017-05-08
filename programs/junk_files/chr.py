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
import scipy.sparse as sparse



f = lambda x: numpy.exp(x)

# Problem setup
# m = 15
# a = 0.0
# b = 2.0 * numpy.pi
# u_a = 0.0
# u_b = 0.0
# f = lambda x: - numpy.sin(x)

def interp(x_fine, x_coarse, f):
    return numpy.interp(x_fine, x_coarse, f)


def solve_iterative(U, f, m, a, b, u_a, u_b):
    # Descretization
    x_bc = numpy.linspace(a, b, m + 2)
    x = x_bc[1:-1]
    delta_x = (b - a) / (m + 1)
    for i in xrange(1, m + 1):
        U[i] = 0.5 * (U[i+1] + U[i-1]) - f(x_bc[i]) * delta_x**2 / 2.0

    return U

def solve_exact(U, f, A, m, a, b, u_a, u_b, rhs=None):
    # Descretization
    x_bc = numpy.linspace(a, b, m + 2)
    x = x_bc[1:-1]
    delta_x = (b - a) / (m + 1)
    
    # # Construct matrix A
    # e = numpy.ones(m)
    # A = sparse.spdiags([e, -2*e, e], [-1,0,1], m, m).tocsr()
    # A /= delta_x**2 

    # Boundary conditions
    if rhs == None:
        B = f(x)
    else:
        B = rhs[1:-1]
    B[0] -= u_a / delta_x**2
    B[-1] -= u_b / delta_x**2 

    # Solve with linalg.solve()
    U[1:-1] = numpy.linalg.solve(A, B)

    return U


def solve_multi(U, f, m, a, b, u_a, u_b, level, max_level=4, rhs = None):
    # Descretization
    x_bc = numpy.linspace(a, b, m + 2)
    x = x_bc[1:-1]
    delta_x = (b - a) / (m + 1)

    # Construct matrix A
    e = numpy.ones(m)
    A = sparse.spdiags([e, -2*e, e], [-1,0,1], m, m).toarray()
    A /= delta_x**2 

    if level == max_level: 
        return solve_exact(U, f, A, m, a, b, u_a, u_b, rhs), x_bc
    
    else: 
        # for j in range(vdown)
        if rhs is None:
            residue = numpy.zeros(m+2)
            residue[1:-1] = f(x) - numpy.dot(A, U[1:-1])
        else:
            residue = numpy.zeros(m+2)
            residue[1:-1] = rhs[1:-1] - numpy.dot(A, U[1:-1])

        ### restriction matrix could be implemented insted of [::2]
        d, junk = solve_multi(numpy.zeros(m/2 +2), f, m/2, a, b, u_a, u_b, level + 1, rhs = residue[::2])
        d = interp(x_bc, x_bc[::2], d)

        U += d

        for j in range(10): ### should be num interations on v_up
            U = solve_iterative(U, f, m, a, b, u_a, u_b)

    return U, x_bc



def main():
     # # Problem setup
    m = 15
    a = 0.0
    b = 1.0
    u_a = 0.0
    u_b = 3.0

    mg_iterations = 2
    u_true = lambda x: (4.0 - numpy.exp(1.0)) * x - 1.0 + numpy.exp(x)
    
    # u_true = lambda x: numpy.sin(x)

    U1, x_bc1 = solve_multi(numpy.zeros(m+2), f, m, a, b, u_a, u_b, 1)
    # error_mg.append(numpy.linalg.norm(u_true(x_bc1) - U1, ord=2))

    U2, x_bc2 = solve_multi(U1[::2], f, m/2, a, b, u_a, u_b, 2)

    U3, x_bc3 = solve_multi(U2[::2], f, m/4, a, b, u_a, u_b, 3)

    U4, x_bc4 = solve_multi(U3[::2], f, m/8, a, b, u_a, u_b, 4)

    # Plot result
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(x_bc1, U1, '-or', label="m=15")
    axes.plot(x_bc2, U2, '-ob', label="m=7")s
    axes.plot(x_bc4, U4, '-og', label="m=1") 
    # axes.plot(x_bc5, U5, '-oy', label="m=3") 
    # axes.plot(x_bc6, U6, '-ob', label="m=7") 
    # axes.plot(x_bc7, U7, '-or', label="m=15") 

    axes.plot(x_bc1, u_true(x_bc1), 'k', label="True")
    axes.set_title("Solution to $u_{xx} = e^x$")
    axes.set_xlabel("x")
    axes.set_ylabel("u(x)")
    axes.legend(loc=2)

    # fig = plt.figure()
    # # fig.set_figwidth(fig.get_figwidth() * 2)
    # axes = fig.add_subplot(1, 1, 1)
    # # axes.semilogy(mg_list, error_mg, 'or')
    # # axes.semilogy(mg_list, no_mg_error, 'ob')
    # # axes.semilogy(range(iterations_J), numpy.ones(iterations_J) * delta_x**2, 'r--')
    # axes.set_title("Subsequent Step Size - J")
    # axes.set_xlabel("Iteration")
    # axes.set_ylabel("$||U^{(k)} - U^{(k-1)}||_2$")
    # # axes = fig.add_subplot(1, 2, 2)
    # # axes.semilogy(range(iterations_J), convergence_J, 'o')
    # # axes.semilogy(range(iterations_J), numpy.ones(iterations_J) * delta_x**2, 'r--')
    # # axes.set_title("Convergence to True Solution - J")
    # # axes.set_xlabel("Iteration")
    # # axes.set_ylabel("$||u(x) - U^{(k-1)}||_2$")

    plt.show()

if __name__ == "__main__":
    main()