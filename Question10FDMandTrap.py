# -*- coding: utf-8 -*-
"""
Created on 12.11.2020

@author: Olav Milian
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import sympy as sym
"""For nicer plotting"""
sym.init_printing()

fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (14, 7), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
            'legend.handlelength': 1.5}
plt.rcParams.update(newparams)

# some constants
G_SC = 1360
A_out = 201.4
B_out = 1.45
s2 = -0.477
au = 0.38
al = 0.68
# k = ?
# R_e = 6.3781e6
# The goal for Q
Q_goal = G_SC / 4
# from the relevant paper
# D
D = 0.649  # k / R_e
# the temperature at the ice-cap
T_s = -10
# the average temperature
T_0 = 14.97


def plotTx(xvec, deltaT_dict, t_end, text, omega):
    # function to make the plot
    # xvec - the x-s values
    # deltaT_dict - dictionary of ([T-values for corresponding t-value], t-value)
    # t_end - last t value
    # text - text for the plot

    # make a figure
    fig = plt.figure()
    # One subplot
    ax = fig.add_subplot(111)
    for key in deltaT_dict:
        Tvec = deltaT_dict[key][0]
        t = deltaT_dict[key][1]
        # plot x, T
        if t == 0:
            ax.plot(xvec, Tvec, label="$\delta T(0,x)$", linewidth=3, color="r", zorder=1e15)
        elif t == t_end:
            ax.plot(xvec, Tvec, label="$\delta T("+ str(t_end)+",x)$", linewidth=3, color="k")
        else:
            ax.plot(xvec, Tvec)

    # get the gid and sett axis labels
    ax.grid()
    ax.set_xlim(0, 1)
    # ax.set_ylim(-50, 50)
    ax.set_xlabel("$x$")
    ax.set_ylabel("Temperature, $^{\circ}C$")
    ax.legend(loc=9, bbox_to_anchor=(0.5, -0.11), ncol=5)
    # plot title
    ax.set_title("$\delta T(t, x)=" + text + "$")
    # adjust
    plt.subplots_adjust(hspace=0.3)

    """Please uncomment to save the plot"""
    # plt.savefig("Q10Omega" + str(omega) +".pdf", bbox_inches='tight')

    # show the plot
    plt.show()

def getA(N):
    # function to get the FDM matrix A
    # N - Number of interior nodes
    # useavg - use the average as "BC", default False

    # the grid size
    h = 1 / (N + 1)
    # define 1/h^2 as a constant
    hh = h * h
    # values of x in the uniform grid
    xvec = np.linspace(0, 1, N + 2)

    # initialize A, b and F
    Nbar = N + 2
    A = sparse.dok_matrix((Nbar, Nbar))

    # D d/dx (1-x^2) dT/dx = D ((1 - x^2) d^2T/dx^2 - 2x dT/dx)
    # Internal nodes
    for i in range(1, N+1):
        xi = xvec[i]
        A[i, i] = D * (1 - xi * xi) * -2 / hh - B_out
        A[i, i+1] = D * ((1 - xi * xi) / hh - xi / h)
        A[i, i-1] = D * ((1 - xi * xi) / hh + xi / h)

    # BC x=0, using ghost point at x=0, 0 = d deltaT/dx = (deltaT_{-1} - deltaT_1}/(2h) => deltaT_{-1} = deltaT_1
    A[0, 0] = D * -2 / hh - B_out
    A[0, 1] = D * 2 / hh

    # BC x=1, backward difference
    # backward difference, (1 - x ^ 2) = 0
    A[N + 1, N + 1] = - 2 * D / h - B_out
    A[N + 1, N] = 2 * D / h

    return A


def Trap(N, M, u0, omega, t_end, savestep=5):
    # function to preform the A-stable implicit trapezoid rule, u = deltaT
    # N - number of internal nodes
    # M - number of time-steps to split interval [0, T] into
    # u0 - function pointer to deltaT at t=0
    # omega - the parameter omega for u0
    # t_end - end time
    # savestep - save values that have j = 10 * m, always save 0 and t_end, default 5

    # get the time - step
    k = t_end / M
    k2 = k / 2

    # values of x in the uniform grid
    xvec = np.linspace(0, 1, N + 2)
    # dictionary to store u
    u_dict = dict()

    # get the FDM matrix A
    A = getA(N)
    # the identity
    I = sparse.identity(A.shape[0])
    # the lhs matrix
    lhs = (I - k2 * A).tocsr()
    # the rhs matrix, B
    B = (I + k2 * A).tocsr()

    # Initial setup
    u_current = u0(xvec, omega)

    # save time-picture
    u_dict[0] = [u_current, 0]
    savecount = 1

    for j in range(1, M + 1):
        # the time
        tk1 = j * k
        # the right hand side vector
        rhs = B @ u_current
        # next u
        u_next = spsolve(lhs, rhs)

        # save time-picture
        if j % savestep == 0 or j == M:
            u_dict[savecount] = [u_next, tk1]
            savecount += 1

        # update
        u_current = u_next

    return u_dict


# number of interior nodes
N = 1000
M = 500
# values of x in the uniform grid
xvec = np.linspace(0, 1, N + 2)

# choosing to test 1/100 * sin(omega * np.pi * x)
text_gen = "\\frac{1}{100}sin("
deltaT0 = lambda x, omega=1: 0.01 * np.sin(omega * np.pi * x)
t_end = 1

omega = 2
deltaT_dict = Trap(N, M, deltaT0, omega, t_end)
text = text_gen + str(omega) + "\pi x)"
plotTx(xvec, deltaT_dict, t_end, text, omega)

omega = 4
deltaT_dict1 = Trap(N, M, deltaT0, omega, t_end)
text = text_gen + str(omega) + "\pi x)"
plotTx(xvec, deltaT_dict1, t_end, text, omega)


















