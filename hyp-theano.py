from scipy import constants as const
from scipy import optimize as opt
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import theano as th
import theano.tensor as T
#import theano.sandbox.linalg.ops as sT
from theano.sandbox.linalg import ops as sT

import GPy

import MLtools

import time

N = 100
D = 2

# Theano routine to create the covariance matrix.
hypsf2 = T.scalar('hypsf2')
hypard = T.vector('hypard')
X = T.matrix('X')

distmat = T.sum(
               ((T.reshape(X, (X.shape[0], 1, X.shape[1]) ) - X) / hypard)**2,
               2)

rbf = hypsf2 * T.exp(-distmat / (2.0))

# Define functions
fdistmat = th.function([X, hypard], distmat)
frbf = th.function([X, hypsf2, hypard], rbf)
fdrbf_dsf2 = th.function([hypsf2], T.jacobian(rbf.flatten(), hypsf2))

curX = rnd.randn(10, 2)
curard = np.array([1, 0.5])

kern = GPy.kern.rbf(2, 1.0, curard, True)
assert np.max(frbf(curX, 1.0, curard) - kern.K(curX)) < 10**-14

print fdrbf_dsf2(1.0)

# You were looking into finding the derivative w.r.t. all the hyperparameters. You'll need to use the function scan for this.
