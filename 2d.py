############################################################
#
# Avi Schwarzschild, Andres Soto
# Numerical Methods for PDE's 
# Final Project
# April 2017
#
############################################################

############
# Working file for 2D problem
############

import numpy
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d

# # Problem setup
# m = 15
# a = 0.0
# b = 1.0
# u_a = 0.0
# u_b = 3.0
# f = lambda x: numpy.exp(x)

# Problem setup
m = 15
a = 0.0
b = 2.0 * numpy.pi
u_a = 0.0
u_b = 0.0

f = lambda x, y: 1.25 * numpy.exp(x + y / 2.0)
u_true = lambda x, y: numpy.exp(x + y / 2.0)

def interp(x_fine, x_coarse, f):
    return numpy.interp(x_fine, x_coarse, f)

def interp2(xc, yc, f):
    n = len(xc[::2])
    x = numpy.repeat(xc[::2], n)
    y = numpy.tile(yc[::2], n)
    f_array = f.reshape(len(f[0])**2, order='F')
    new_f = interp2d(x, y, f)

    return new_f(xc, yc).T



########
# Adapted from 06_iterative.ipynb by Prof. Kyle Mandli
########

def solve_jacobi(X, Y, alpha_x, beta_x, alpha_y, beta_y, f, iterations_J=None, U_0=None):
    # Descritization, extract x, y arrays, and delta x, delta y arrays, m and n
    x = X[:, 0]
    y = Y[0, :]
    dx = x[1] - x[0]
    m = len(x) - 2  

    # Expected iterations needed
    if iterations_J == None:
        iterations_J = int(2.0 * numpy.log(delta_x) / numpy.log(1.0 - 0.5 * numpy.pi**2 * delta_x**2))

    if U_0 != None:
        U_new = U_0
        if m < len(U_0[:,0]) - 2:
            U_new = U_new[::2, ::2]
        if m > len(U_0) - 2:
            U_new = interp2(x, y, U_0)
    else:
        U_new = numpy.zeros((m + 2, m+2))
        U_new[0, :] = alpha_x(y)
        U_new[-1, :] = beta_x(y)
        U_new[:, 0] = alpha_y(x)
        U_new[:, -1] = beta_y(x)

    # convergence_J = numpy.zeros(iterations_J)
    # step_size_J = numpy.zeros(iterations_J)
    for k in xrange(iterations_J):
        U = U_new.copy()
        for i in xrange(1, m + 1):
            for j in xrange(1, m+1):
                U_new[i, j] = 0.25 * (U[i+1, j] + U[i-1, j] + U[i, j-1] + U[i, j+1]) - f(x[i], y[j]) * dx**2 / 4.0

        # step_size_J[k] = numpy.linalg.norm(U - U_new, ord=2)
        # convergence_J[k] = numpy.linalg.norm(u_true(x_bc) - U_new, ord=2)

    return U_new, X, Y


def main():
    mg_list = [1,2,4,10,20, 40, 80, 1000]
    # mg_iterations = 2
    for mg_iterations in mg_list:
        # delta_x = 1.0 / (m + 1)
        # delta_y = 2.0 / (n + 1)
        x = numpy.linspace(0.0, 1.0, m + 2)
        y = numpy.linspace(0.0, 1.0, m + 2)
        X, Y = numpy.meshgrid(x, y)
        # Transpose these so that the coordinates match up to (i, j)
        X = X.transpose()
        Y = Y.transpose()    
        u_true = lambda x, y: numpy.exp(x + y / 2.0)

            # Boundary Functions
        alpha_x = lambda y: numpy.exp(y/2.0)
        beta_x = lambda y: numpy.exp(1.0 + y/2.0)
        alpha_y = lambda x: numpy.exp(x)
        beta_y = lambda x: numpy.exp(x + 0.5)


        
        error_mg = []

        U1, X1, Y1 = solve_jacobi(X, Y, alpha_x, beta_x, alpha_y, beta_y, f, mg_iterations)

        U2, X2, Y2 = solve_jacobi(X[::2, ::2], Y[::2, ::2], alpha_x, beta_x, alpha_y, beta_y, f, mg_iterations, U1.copy())
        
        U3, X3, Y3 = solve_jacobi(X2[::2, ::2], Y2[::2, ::2], alpha_x, beta_x, alpha_y, beta_y, f, mg_iterations, U2.copy())
        
        U4, X4, Y4 = solve_jacobi(X3[::2, ::2], Y3[::2, ::2], alpha_x, beta_x, alpha_y, beta_y, f, mg_iterations, U3.copy())
        
        U5, X5, Y5 = solve_jacobi(X3, Y3, alpha_x, beta_x, alpha_y, beta_y, f, mg_iterations, U4.copy())

        U6, X6, Y6 = solve_jacobi(X2, Y2, alpha_x, beta_x, alpha_y, beta_y, f, mg_iterations, U5.copy())

        U7, X7, Y7 = solve_jacobi(X1, Y1, alpha_x, beta_x, alpha_y, beta_y, f, mg_iterations, U6.copy())



        # U_list = [U1 ,U2 ,U3 ,U4 ,U5 ,U6 ,U7]
        # X_list = [X1 ,X2 ,X3 ,X4 ,X5 ,X6 ,X7]
        # Y_list = [Y1 ,Y2 ,Y3 ,Y4 ,Y5 ,Y6 ,Y7]

        U_list = [U1, U7]
        X_list = [X1, X7]
        Y_list = [Y1, Y7]

        plot_list = zip(U_list, X_list, Y_list)
        # for u, x, y in plot_list:

            # fig = plt.figure()
            # fig.set_figwidth(fig.get_figwidth())
            # axes = fig.add_subplot(1, 2, 1, aspect='equal')
            # plot = axes.pcolor(x, y, u, vmax=7.0, vmin=0.0, cmap=plt.get_cmap("Blues"))
            # fig.colorbar(plot, label="$U$")
            # axes.set_title("Computed Solution")
            # axes.set_xlabel("x")
            # axes.set_ylabel("y")
            # axes = fig.add_subplot(1, 2, 2, aspect='equal')
            # plot = axes.pcolor(x, y, u_true(x, y), vmax=7.0, vmin=0.0, cmap=plt.get_cmap("Blues"))
            # fig.colorbar(plot, label="$u(x,t)$")
            # axes.set_title("True Solution")
            # axes.set_xlabel("x")
            # axes.set_ylabel("y")

        U1, X1, Y1 = solve_jacobi(X, Y, alpha_x, beta_x, alpha_y, beta_y, f, 7*mg_iterations)

        # fig = plt.figure()
        # fig.set_figwidth(fig.get_figwidth())
        # axes = fig.add_subplot(1, 2, 1, aspect='equal')
        # plot = axes.pcolor(x, y, u, vmax=7.0, vmin=0.0, cmap=plt.get_cmap("Blues"))
        # fig.colorbar(plot, label="$U$")
        # axes.set_title("Computed Solution")
        # axes.set_xlabel("x")
        # axes.set_ylabel("y")
        # axes = fig.add_subplot(1, 2, 2, aspect='equal')
        # plot = axes.pcolor(x, y, u_true(x, y), vmax=7.0, vmin=0.0, cmap=plt.get_cmap("Blues"))
        # fig.colorbar(plot, label="$u(x,t)$")
        # axes.set_title("True Solution")
        # axes.set_xlabel("x")
        # axes.set_ylabel("y")

        plt.show()

        error = numpy.linalg.norm(U1 - u_true(X1, Y1), ord=2)
        print error

        error = numpy.linalg.norm(U7 - u_true(X7, Y7), ord=2)
        print error

if __name__ == "__main__":
    main()