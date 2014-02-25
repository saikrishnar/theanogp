from scipy import constants as const
from scipy import optimize as opt
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import theano
import theano.tensor as T
import theano.sandbox.linalg.ops as sT
from theano.sandbox.linalg import ops as sT

import matplotlib
import matplotlib.pyplot as plt

import GPy

# import MLtools

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
fdistmat = theano.function([X, hypard], distmat)
frbf = theano.function([X, hypsf2, hypard], rbf)

# drbf_dsf2 = T.jacobian(rbf.flatten(), hypsf2)
# fdrbf_dsf2 = theano.function([X, hypsf2, hypard], drbf_dsf2)
drbf_dsf2, u = theano.scan(lambda i, y, x: T.jacobian(y[i], x), sequences=T.arange(rbf.shape[0]), non_sequences=[rbf, hypsf2])
fdrbf_dsf2 = theano.function([X, hypsf2, hypard], drbf_dsf2, updates=u)

# drbf_dard = T.jacobian(rbf.flatten(), hypard)
drbf_dard, u = theano.scan(lambda i, y, x: T.jacobian(y[i], x), sequences=T.arange(rbf.shape[0]), non_sequences=[rbf, hypard])
fdrbf_dard = theano.function([X, hypsf2, hypard], drbf_dard, updates=u)
print u

# JX, updatesX = theano.scan(lambda i, y, x: T.jacobian(y[i], x), sequences=T.arange(Z.shape[0]), non_sequences=[Z, X])


curX = rnd.randn(10, 2)
curard = np.array([1, 0.5])

kern = GPy.kern.rbf(2, 1.0, curard, True)
assert np.max(frbf(curX, 1.0, curard) - kern.K(curX)) < 10**-14

dsf2 = fdrbf_dsf2(curX, 1.0, curard)
dard = fdrbf_dard(curX, 1.0, curard)

# print dsf2
print dard.shape

# You were looking into finding the derivative w.r.t. all the hyperparameters. You'll need to use the function scan for this.


# Plot stuff
K = frbf(curX, 1.0, curard)
dK_dsf2 = fdrbf_dsf2(curX, 1.0, curard)

plt.ion()
# plt.imshow(K.reshape(10, 10), interpolation='Nearest')
# plt.figure()
plt.imshow(dK_dsf2, interpolation='Nearest')
plt.figure()
plt.imshow(dard[:, :, 0], interpolation='Nearest')
plt.figure()
plt.imshow(dard[:, :, 1], interpolation='Nearest')
plt.show(block=True)

