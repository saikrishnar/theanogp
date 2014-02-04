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

mu = T.vector('mu')
sigma = T.matrix('sigma')
prec = sT.matrix_inverse(sigma)

# x = np.array([[15, -1.5], [-1.5, 1.5], [-1.4, 1.5], [1.4, -1.5], [-45.0, 83.5]])
# x = np.array([[15, -1.5],
#               [-1.5, 1.5],
#               [-1.4, 1.5],
#               [1.4, -1.5],
#               [-45.0, 83.5],
#               [-100.3, 68.3],
#               [1000.4, 432.4],
#               [32441.8, 12341.3]])

N = 100

x = rnd.randn(N, 1)

D = x.shape[0]
d = x - mu

p = T.log(T.prod((2*const.pi)**(-0.5*D) * sT.det(sigma)**-0.5 *
     T.exp(sT.diag(-0.5*T.dot(d, T.dot(prec, d.T))))))

p1 = T.sum(-0.5*D*T.log(2*const.pi) +
           -0.5*T.log(sT.det(sigma)) +
           -0.5*sT.diag(T.dot(d, T.dot(prec, d.T)))
          )

p2 = T.sum(-0.5*D*T.log(2*const.pi) +
           -T.sum(T.log(sT.diag(sT.cholesky(sigma)))) +
           -0.5*sT.diag(T.dot(d, T.dot(prec, d.T)))
          )

fp = th.function([mu, sigma], p)
fp1 = th.function([mu, sigma], p1)
fp2 = th.function([mu, sigma], p2)

# GP inputs, N points, 2D
z = rnd.randn(N, 2)
kern = GPy.kern.rbf(2) + GPy.kern.white(2, 10**-6)
curmu = np.zeros(N)
cursig = kern.K(z)

# stabdet = 2*T.sum(T.log(sT.diag(sT.cholesky(sigma))))
# fdet = th.function([sigma], stabdet)
# print fdet(cursig)
# print la.slogdet(cursig)

# Compare to multivalued normal
print 'MLtools calculation:'
print np.sum(MLtools.mvnlogpdf(x, curmu, cursig))
print 'Theano calculations:'
print fp(curmu, cursig)
print fp1(curmu, cursig)
print fp2(curmu, cursig)

