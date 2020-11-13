# -*- coding: utf-8 -*-
"""
Created on 11.11.2020

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
D = 0.649 #k / R_e
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
def getAbF(N, x_s):
    # the grid size
    h = 1 / (N + 1)
    # define 1/h^2 as a constant
    hh = h * h
    # values of x in the uniform grid
    xvec = np.linspace(0, 1, N + 2)
    # initialize A and b
    A = sparse.dok_matrix((N+2, N+2))
    b = - A_out * np.ones(N+2)
    F = a(xvec, x_s) * S(xvec)

    # Internal nodes
    for i in range(1, N+1):
        xi = xvec[i]
        A[i, i] = D * (1 - xi * xi) * 2 / hh + B_out
        A[i, i+1] = - D * ((1 - xi * xi) / hh + xi / h)
        A[i, i-1] = - D * ((1 - xi * xi) / hh - xi / h)

    # BC x=0
    A[0, 0] = - 1 / h
    A[0, 1] = 1 / h
    b[0] = 0
    F[0] = 0

    # BC x=1
    A[N+1, N+1] = - D / h + B_out
    A[N+1, N] = D / h
    # F[N+1] = a(1, x_s) * S(1)
    return A, b, F

def Solve_bvp(N, x_s, Q):
    # the x - vector
    xvec = np.linspace(0, 1, N + 2)
    # get A, b and F
    A, b, F = getAbF(N, x_s)
    # solve the system for given Q
    Tvec = spsolve(A.tocsr(), b + Q_goal * F)

    return xvec, Tvec


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

def Q_finder(N, x_s):
    # get A, b, F
    A, b, F = getAbF(N, x_s)
    # the step size
    h = 1 / (N + 1)
    # values of x in the uniform grid
    xvec = np.linspace(0, 1, N + 2)
    # index of the largest x smaller than x_s
    k = int(x_s / h)
    # call it xk
    xk = xvec[k]
    xk1 = xvec[k + 1]


    Abar = sparse.lil_matrix((N + 3, N + 3))
    # insert A into Abar
    Abar[:-1, :-1] = A
    # insert F into Abar
    Abar[:-1, -1] = - F.reshape((N + 2, 1))  # make column-vector of the row-vector

    Abar[-1, k] = (xk1 - x_s) / h
    Abar[-1, k+1] = (x_s - xk) / h
    Fbar = np.zeros(N+3)
    Fbar[:-1] = b
    Fbar[-1] = -10

    TQvec = spsolve(Abar.tocsr(), Fbar)
    Q = TQvec[-1]
    Tvec = TQvec[:-1]
    print(Q, Q_goal, abs(Q - Q_goal))
    print((xk1 - x_s) / h * TQvec[k] + (x_s - xk) / h * TQvec[k+1])

    # Tvec = spsolve(A.tocsr(), b + Q * F)
    plotTx(xvec, Tvec, x_s)

    Tvec = spsolve(A.tocsr(), b + Q_goal * F)
    plotTx(xvec, Tvec, x_s)

Q_finder(N=500, x_s=0.95)











# number of interior nodes
N = 500
# position of ice cap.
x_s = 0.77












