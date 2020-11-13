# -*- coding: utf-8 -*-
"""
Created on 05.11.2020

@author: Olav Milian
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import legendre

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

def Hn(x_s, n):
    f = lambda x: a(x, x_s) * S(x) * legendre(n)(x)
    h = (2 * n + 1) * (quad(f, 0, x_s)[0] + quad(f, x_s, 1)[0])
    return h

def CN(x_s, N):
    c = np.sum([Hn(x_s, n) * legendre(n)(x_s)/ ((n * (n + 1) * D + B_out)) for n in range(N+1)])
    return c

def QN(x_s, N):
    q = (A_out + B_out * T_s) / (B_out * CN(x_s, N))
    return q

xs = 0.75
Nvec = np.arange(10, 20)
#Qvec = np.array([QN(xs, N) for N in Nvec])
#Q50 = QN(xs, 20)

#print(Qvec, Q50)

xvec = np.linspace(0.001, 1, 61)
Q2 = np.array([QN(x, 25) for x in xvec])

print(Q2, Q_goal)

arg = np.argmin(np.abs(Q2-Q_goal))

print(xvec[arg], Q2[arg], Q_goal, np.arcsin(xvec[arg]))







