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

def plotTx(xvec, Tvec, x_s):
    plt.figure()

    plt.plot(xvec, Tvec, label="$T(x)$")
    # plt.plot(x_s, T_s, 'ro', label="$(x_s, T_s)$")
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

# function to get the FDM matrix A
def getAbk(N, x_s, eps=1e-10):
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

    if abs(xk - x_s) <= eps:
        print("It is assumed that x_s is not a grid point, but it is, adding a epsilon to x_s")
        while abs(xk - x_s) <= eps:
            x_s += eps
            if x_s > 1:
                x_s -= 2 * eps
            k = int(x_s / h)
            # call it xk
            xk = xvec[k]
        print("Using x_s =", x_s)

    xk1 = xvec[k + 1]


    def Indexmap(i):
        if i <= k:
            return i
        else:
            return i + 1

    # initialize A and b
    A = sparse.dok_matrix((N+4, N+4))
    b = - A_out * np.ones(N+4)
    F = np.zeros(N+3)
    # F = a(xvec, x_s) * S(xvec)

    # -D d/dx (1-x^2) dT/dx = -D ((1 - x^2) d^2T/dx^2 - 2x dT/dx)
    # Internal nodes
    for i in range(1, N+1):
        index = Indexmap(i)
        xi = xvec[index]
        if index not in (k, k+1, k+2):
            A[index, index] = D * (1 - xi * xi) * 2 / hh + B_out
            A[index, index+1] = - D * ((1 - xi * xi) / hh - xi / h)
            A[index, index-1] = - D * ((1 - xi * xi) / hh + xi / h)
            F[index] = a(xi, x_s) * S(xi)

    # BC x=0
    A[0, 0] = - 1 / h
    A[0, 1] = 1 / h
    b[0] = 0
    F[0] = 0

    # BC x=x_s
    A[k+1, k+1] = 1
    b[k+1] = T_s

    # BC x=1
    A[N+2, N+2] = D / h + B_out
    A[N+2, N+1] = - D / h
    F[N+2] = a(1, x_s) * S(1)

    # use  irregular grid
    # -D d/dx (1-x^2) dT/dx = -D ((1 - x^2) d^2T/dx^2 - 2x dT/dx)
    # find eta s.t x_s = xk + eta * h = xk1 - (1 - eta) * h
    eta = (x_s - xk) / h  # 0 < eta < 1, eta = 0 handled above, and also eta=1, since than xk is xk1 and xk1 is xk2.
    # for xk
    A[k, k] = D * ((1 - xk * xk) * 2 / (hh * eta)) + B_out
    A[k, k - 1] = - D * ((1 - xk * xk) * 2 / (hh * (1 + eta)) + 2 * xk / (h * (1 + eta)))
    # T_s
    A[k, k + 1] = - D * ((1 - xk * xk) * 2 / (hh * eta * (1 + eta)) - 2 * xk / (h * (1 + eta)))
    F[k] = a(xk, x_s) * S(xk)

    # for xk1
    mu = 1 - eta
    # -D d/dx (1-x^2) dT/dx = -D ((1 - x^2) d^2T/dx^2 - 2x dT/dx)
    if k < N:  # we have the node (xk2, Tk2)
        A[k+2, k+2] = D * ((1 - xk1 * xk1) * 2 / (hh * mu)) + B_out
        A[k+2, k+3] = - D * ((1 - xk1 * xk1) * 2 / (hh * (1 + mu)) - 2 * xk1 / (h * (1 + mu)))
        A[k+2, k+1] = - D * ((1 - xk1 * xk1) * 2 / (hh * mu * (1 + mu)) + 2 * xk1 / (h * (1 + mu)))
        F[k+2] = a(xk1, x_s) * S(xk1)

    else:  # the node (xk1, Tk1) is the end-node, and (1^2-1)=0
        A[k+2, k+2] = D / (mu * h) + B_out
        A[k+2, k+1] = - D / (mu * h)
        # F done above for x=1

    # insert F into A
    A[:-1, -1] = - F.reshape((N + 3, 1))  # make column-vector of the row-vector

    # Now BC x=x_S on the derivative, factored out h
    A[N+3, k] = - 1 / eta  # Tk
    A[N+3, k+1] = 1 / eta - 1 / mu  # Ts
    A[N+3, k+2] = 1 / mu  # Tk1
    b[N+3] = 0

    return A, b, F, k


def get_xvec(N, x_s, k):
    x_vec = np.linspace(0, 1, N + 2)
    argu = np.nonzero(x_vec > x_s)[0]
    argl = np.nonzero(x_vec < x_s)
    xvec = np.zeros(N + 3)
    xvec[argu + 1] = x_vec[argu]
    xvec[k + 1] = x_s
    xvec[argl] = x_vec[argl]
    return xvec

def TQsolver(N, x_s):
    A, b, F, k = getAbk(N, x_s)
    xvec = get_xvec(N, x_s, k)
    Tvec = spsolve(A[:-1, :-1].tocsr(), b[:-1] + Q_goal * F)
    plotTx(xvec, Tvec, x_s)
    TQvec = spsolve(A.tocsr(), b)
    # print(TQvec, TQvec[k+1])
    Tvec = TQvec[:-1]
    Q = TQvec[-1]
    plotTx(xvec, Tvec, x_s)

    print(A[0], b[0], "hei")
    print("-"*40)
    print(A[k + 1], b[k + 1], "hei")
    print("-" * 40)
    print(A[k], b[k], "hei")
    print("-" * 40)
    print(A[k + 2], b[k + 2], "hei")
    print("-" * 40)
    print(A[N+3], b[N+3], "hei")
    print("-" * 40)
    print(A[2], b[2], "hei")
    print(Q, Q_goal, abs(Q - Q_goal))
    print(N+3, A.shape)






# number of interior nodes
N = 1000
# position of ice cap.
x_s = 0.75
TQsolver(N, x_s)


"""Backwars diff = forward diff
Ts = -10
A[s, s] = 1, b[s] = -10
(Tk - Ts) / eta -  (Ts - Tk1) / mu = 0 , mu = 1 - eta"""






