# -*- coding: utf-8 -*-
"""
Created on 11.11.2020

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


# the co-albedo
def a(xvec, x_s):
    # xvec - array of x values
    # x_s - parameter for ice-cap location
    if isinstance(xvec, (int, float)):
        xvec = np.asarray([xvec])

    # initialize the output vector
    avec = np.zeros_like(xvec)
    # indices
    argu = np.nonzero(xvec > x_s)
    argl = np.nonzero(xvec < x_s)
    args = np.nonzero(xvec == x_s)
    # set the values
    avec[argu] = au
    avec[argl] = al
    avec[args] = (au + al) / 2
    return avec

# the annual average
S = lambda x: 1 + s2 * (3 * x * x - 1) / 2

def plotTx(xvec, Tvec, x_s):
    # function to make the plot
    # xvec - the x-s values
    # Tvec - the T-values
    # x_s - position of the ice-cap

    # make a figure
    fig = plt.figure()
    # One subplot
    ax = fig.add_subplot(111)
    # plot x, T
    ax.plot(xvec, Tvec, label="$T(x)$")
    # the point (x_s,T_s)
    ax.plot(x_s, T_s, 'ro', label="$(x_s, T_s)$")
    # lines x_s and T_s
    ax.vlines(x_s, ymin=-100, ymax=200, linestyles='--', color='k', label="$x_s="+"{:.2f}".format(x_s)+"$")
    ax.hlines(T_s, xmin=0, xmax=1, linestyles='--', color='k', label='$T_s=-10^{\circ}C$')
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
    ax.set_title("Temperature, $T(x)$")
    # adjust
    plt.subplots_adjust(hspace=0.3)

    """Please uncomment to save the plot"""
    # plt.savefig("TQwithx_s0" + str(int(round(x_s, 2) * 100)) + ".pdf", bbox_inches='tight')

    # show the plot
    plt.show()


def getAbF(N, x_s, ifprint=True, useavg=True):
    # function to get the FDM matrix A
    # N - Number of interior nodes
    # x_s - position of the ice-cap, x_S will be restricted to be on the grid.
    # ifprint - pint info to user, default True
    # useavg - use the average as "BC", default True

    # the grid size
    h = 1 / (N + 1)
    # define 1/h^2 as a constant
    hh = h * h
    # values of x in the uniform grid
    xvec = np.linspace(0, 1, N + 2)
    # index of the largest x smaller than x_s
    k = int(x_s / h)
    # call it xk
    xk = xvec[k]
    xk1 = xvec[k + 1]

    # check if x_s is one the grid
    if xk != x_s:
        if ifprint:
            print("Assuming that x_s is one the grid, using closet value on grid as x_sk")
        arg = np.argmin(np.abs(np.array([xk, xk1]) - x_s))
        if arg == 0:
            # xk is the closest  to x_s
            x_sk = xk
        elif arg == 1:
            # xk1 is the closest to x_s
            x_sk = xk1
            # update index k
            k = k + 1
        if ifprint:
            print("Using x_s = ", x_sk)
    else:
        x_sk = x_s

    # initialize A, b and F
    Nbar = N + 3
    A = sparse.dok_matrix((Nbar, Nbar))
    b = - A_out * np.ones(Nbar)
    F = a(xvec, x_sk) * S(xvec)

    # -D d/dx (1-x^2) dT/dx = -D ((1 - x^2) d^2T/dx^2 - 2x dT/dx)
    # Internal nodes
    for i in range(1, N+1):
        if i != k:
            xi = xvec[i]
            A[i, i] = D * (1 - xi * xi) * 2 / hh + B_out
            A[i, i+1] = - D * ((1 - xi * xi) / hh - xi / h)
            A[i, i-1] = - D * ((1 - xi * xi) / hh + xi / h)

    # BC x=0
    A[0, 0] = - 1 / h
    A[0, 1] = 1 / h
    b[0] = 0
    F[0] = 0

    # BC x=x_s
    A[k, k] = 1
    b[k] = T_s
    F[k] = 0

    # BC x=1
    if not useavg:
        A[N+1, N+1] = D / h + B_out
        A[N+1, N] = - D / h
    # enforce that the average is T_0
    if useavg:
        A[N+1, :-1] = np.ones(N+2) / (N+2)
        b[N+1] = T_0
        F[N+1] = 0

    # insert F into A
    A[:-1, -1] = - F.reshape((N + 2, 1))  # make column-vector of the row-vector

    # Now BC x=x_S on the derivative, factored out h
    A[N+2, [k-1, k, k+1]] = [-1, 2, -1]
    b[N+2] = 0

    return A, b, F, k, x_sk


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


def xs_finder(N, Nxs, xsmin=0.001, xsmax=0.999, useavg=True):
    # function to find x_s that gives Q(x_s) = Q_goal
    # function gives the best the x_s that minimizes abs(Q(x_s)-Q_goal) from the sample x_s-s
    # N - number of internal nodes in discretion
    # Nxs - number of x_s-s samples
    # xsmin - minimum x_s sample value, default 0.001
    # xsmax - maximum x_s sample value, default 0.999
    # useavg - use the average as "BC", default True

    # the sample x_s-s
    xs_vec = np.linspace(xsmin, xsmax, Nxs)
    # 0 vector for Q-values and found x_S values on grid
    Qvec = np.zeros(Nxs)
    xsk_vec = np.zeros(Nxs)
    for i in range(Nxs):
        # find Q and x_s on the grd and save them in the vectors
        # we will not make plots or print info to user here
        Tvec, Q, x_sk = TQsolver(N, xs_vec[i], ifprint=False, ifplot=False, useavg=useavg)
        Qvec[i] = Q
        xsk_vec[i] = x_sk
    # find index of best Q, also index of the x_s
    arg = np.argmin(np.abs(Qvec - Q_goal))
    # the best values
    Q_best = Qvec[arg]
    xsk_best = xsk_vec[arg]
    # info to user
    print("x_s = ", xsk_best, "gives Q = ", Q_best, "Q_qoal = ", Q_goal, "Absolute difference ", abs(Q_best- Q_goal))
    # return the best x_s on the grid.
    return xsk_best


# number of interior nodes
N = 1000
# number of x_s-s to test
M = 250
# min, max for x_s
xsmin = 0.33
xsmax = 0.40
# position of ice cap.
x_s = xs_finder(N, M, useavg=False)
Tvec, Q, x_sk = TQsolver(N, x_s, ifprint=False, useavg=False)
print("-"*40)
# position of ice cap.
x_s = xs_finder(N, M, useavg=True)
Tvec2, Q2, x_sk2 = TQsolver(N, x_s)








