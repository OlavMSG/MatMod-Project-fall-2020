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


def plotTx(xvec, Tvec, t):
    # function to make the plot
    # xvec - the x-s values
    # Tvec - the T-values
    # t - the time

    # make a figure
    fig = plt.figure()
    # One subplot
    ax = fig.add_subplot(111)
    # plot x, T
    ax.plot(xvec, Tvec, label="$T(x)$")
    # plot the mean line
    T_mean = np.mean(Tvec)
    ax.hlines(T_mean, xmin=0, xmax=1, linestyles='--', color='g',
               label='$T_{avg}='+'{:.2f}'.format(T_mean)+'^{\circ}C$')
    # get the gid and sett axis labels
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("$x$")
    ax.set_ylabel("Temperature, $^{\circ}C$")
    ax.legend(loc=9, bbox_to_anchor=(0.5, -0.11), ncol=5)
    # plot title
    ax.set_title("Temperature, $\delta T(x)$")
    # adjust
    plt.subplots_adjust(hspace=0.3)

    """Please uncomment to save the plot"""
    # plt.savefig("TQwithx_s0" + str(int(round(x_s, 2) * 100)) + ".pdf", bbox_inches='tight')

    # show the plot
    plt.show()


def getA(N, useavg=False):
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


def TQsolver(N, x_s, ifprint=True, ifplot=True, useavg=True):
    # function to solve for T and Q, and ifplot=True pmake a plot
    # N - Number of interior nodes
    # x_s - position of the ice-cap
    # ifprint - pint info to user, default True
    # ifplot - make a plot of x and T, default True
    # useavg - use the average as "BC", default True

    # get A, b, F, k, x_sk
    A, b, F, k, x_sk = getAbF(N, x_s, ifprint, useavg)
    # the x-vector
    xvec = np.linspace(0, 1, N + 2)
    # solve for T and Q
    TQvec = spsolve(A.tocsr(), b)
    # T values are the first values, Q is the last
    Tvec = TQvec[:-1]
    Q = TQvec[-1]
    if ifprint:
        print("x_s = ", x_sk, "gives Q = ", Q, "Q_qoal = ", Q_goal, "Absolute difference ", abs(Q - Q_goal))
    if ifplot:
        plotTx(xvec, Tvec, x_sk)
    return Tvec, Q, x_sk





# number of interior nodes
N = 1000

Tvec, Q, x_sk = TQsolver(N, x_s, ifprint=False, useavg=False)
print("-"*40)
# position of ice cap.
Tvec2, Q2, x_sk2 = TQsolver(N, x_s)
















