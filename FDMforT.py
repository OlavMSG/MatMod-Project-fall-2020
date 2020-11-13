# -*- coding: utf-8 -*-
"""
Created on 04.11.2020

@author: Olav Milian
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, lsqr

# some constants
G_SC = 1360
A_out = 201.4
B_out = 1.45
s2 = -0.477
au = 0.38
al = 0.68
k = 0.34 #OBS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
R_e = 6.3781e6
D = k / R_e
T_s = -10
# The goal for Q
Q_goal = G_SC / 4

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

# function to get the FDM matrix A
def getAbF(N, x_s, eps=1e-14):
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

    # initialize A and b
    A = sparse.dok_matrix((N+2, N+2))
    b = - A_out * np.ones(N+2)
    F = a(xvec, x_s) * S(xvec)

    # Internal nodes
    for i in range(1, N+1):
        xi = xvec[i]
        if i not in (k, k+1):
            A[i, i] = D * (1 - xi * xi) * 2 / hh + B_out
            A[i, i+1] = - D * ((1 - xi * xi) / hh + xi / h)
            A[i, i-1] = - D * ((1 - xi * xi) / hh - xi / h)

    # BC x=0
    A[0, 0] = - 1 / h
    A[0, 1] = 1 / h
    b[0] = 0
    F[0] = 0

    # BC x=1
    A[N+1, N+1] = - 1 / h + B_out
    A[N+1, N] = 1 / h
    # F[N+1] = a(1, x_s) * S(1)

   # For x_s
    if abs(xk - x_s) <= eps:  # x_s = xk, x_s is a grid point
        # for xk = x_s
        # print("x_s on grid")
        A[k, k] = 1
        b[k] = T_s
        F[k] = 0
        # for xk1
        if k < N + 1:
            A[k+1, k+1] = D * (1 - xk1 * xk1) * 2 / hh + B_out
            A[k+1, k + 2] = - D * ((1 - xk1 * xk1) / hh + xk1 / h)
            A[k+1, k] = - D * ((1 - xk1 * xk1) / hh - xk1 / h)
    else: # use  irregular grid
        # find eta s.t x_s = xk + eta * h = xk1 - (1 - eta) * h
        eta = (x_s - xk) / h  # 0 < eta < 1, eta = 0 handled above, and also eta=1, since than xk is xk1 and xk1 is xk2.
        # for xk
        A[k, k] = D * ((1 - xk * xk) * 2 / (hh * eta)) + B_out
        A[k, k - 1] = - D * ((1 - xk * xk) * 2 / (hh * (1 + eta)) - 2 * xk / (h * (1 + eta)))
        # adding value for T_s to b
        b[k] += D * ((1 - xk * xk) * 2 / (hh * eta * (1 + eta)) + 2 * xk / (h * (1 + eta))) * T_s
        # for xk1
        mu = 1 - eta

        if k < N:  # we have the node (xk2, Tk2)
            A[k + 1, k + 1] = D * ((1 - xk1 * xk1) * 2 / (hh * mu)) + B_out
            A[k+1, k+2] = - D * ((1 - xk1 * xk1) * 2 / (hh * (1 + mu)) + 2 * xk1 / (h * (1 + mu)))
            # adding value for T_s to b
            b[k+1] += D * ((1 - xk1 * xk1) * 2 / (hh * mu * (1 + mu)) - 2 * xk1 / (h * (1 + mu))) * T_s
        else: # the node (xk1, Tk1) is the end-node, and (1^2-1)=0
            A[k+1, k+1] = - 1 / (mu * h)
            # adding value for T_s to b
            b[k+1] += - 1 / (mu * h) * T_s

    return A, b, F

def Solve_bvp(N, x_s, Q):
    # the x - vector
    xvec = np.linspace(0, 1, N + 2)
    # get A, b and F
    A, b, F = getAbF(N, x_s)
    # solve the system for given Q
    Tvec = spsolve(A.tocsr(), b + Q_goal * F)
    # make plot
    plotTx(xvec, Tvec, x_s)

def plotTx(xvec, Tvec, x_s):
    plt.figure()

    plt.plot(xvec, Tvec, label="$T(x)$")
    plt.plot(x_s, T_s, 'ro', label="$(x_s, T_s)$")
    plt.hlines(-10, xmin=0, xmax=1, linestyles='--', color='k', label='$T_s=-10^{\circ}C$')
    T_mean = np.mean(Tvec)
    plt.hlines(T_mean, xmin=0, xmax=1, linestyles='--', color='g',
               label='$T_{avg}=' + '{:.2f}'.format(T_mean) +'^{\circ}C$')
    plt.grid()
    plt.xlim(0, 1)
    plt.xlabel("$x$")
    plt.ylabel("Temperature, $^{\circ}C$")
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.11), ncol=4)
    plt.title("Temperature, $T(x)$")
    plt.show()


def Least_square(N, x_s):
    # the x - vector
    xvec = np.linspace(0, 1, N + 2)
    # get A, b and F
    A, b, F = getAbF(N, x_s)

    Abar = sparse.lil_matrix((N+2, N+3))
    # insert A into Abar
    Abar[:, :-1] = A
    # insert F into Abar
    Abar[:, -1] = - F.reshape((N+2, 1))  # make column-vector of the row-vector

    LSQR = lsqr(Abar, b)
    TQvec = LSQR[0]
    res = LSQR[3]
    Q = TQvec[-1]
    return Q, res

def xs_finder(N, Ntest, xsmin=0.001, xsmax=0.999):

    xsvec = np.linspace(xsmin, xsmax, Ntest)
    Qvec = np.zeros(Ntest)
    resvec = np.zeros(Ntest)
    for i in range(Ntest):
        Qvec[i], resvec[i] = Least_square(N, xsvec[i])

    argb = np.argmin(np.abs(Qvec - Q_goal))

    xsb = xsvec[argb]

    print(xsb, Qvec[argb], Q_goal, resvec[argb])

    return xsb








# number of interior nodes
N = 500
# position of ice cap.
x_s = 0.77
Solve_bvp(N, x_s, Q_goal)

xsb1 = xs_finder(N, 10)







