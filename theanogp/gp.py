import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg

import scipy.optimize as opt
import scipy.constants as const

import theano
import theano.tensor as T
from theano.sandbox.linalg import ops as sT

import mltools.simple_optimise as mlopt

class gp(object):
    def __init__(self, kernel, X=None, Y=None):
        self.kernel = kernel
        self.X = X
        self.Y = Y

        self.th_hyp = self.kernel.th_hyp
        self.th_X = self.kernel.th_X
        self.th_N = self.kernel.th_N
        self.th_D = self.kernel.th_D
        self.th_K = self.kernel.th_K

        self.th_Y = T.matrix('Y')

        prec = sT.matrix_inverse(self.th_K)
        # T.sum(T.log(sT.diag(sT.cholesky(sigma))))
        # self.th_lml = (- 0.5 * sT.trace(T.dot(self.th_Y.T, T.dot(prec, self.th_Y))) +
        #                - T.sum(T.log(sT.diag(sT.cholesky(self.th_K)))) +
        #                - 0.5 * self.th_N * T.log(2.0 * const.pi) )
        self.th_lml = (- 0.5 * sT.trace(T.dot(self.th_Y.T, T.dot(prec, self.th_Y))) +
                       - 0.5 * T.log(sT.det(self.th_K)) +
                       - 0.5 * self.th_N * T.log(2.0 * const.pi) )
        self.th_dlml_dhyp = theano.grad(self.th_lml, self.th_hyp)

        self.lml = theano.function([self.th_hyp, self.th_X, self.th_Y], self.th_lml)
        self.dlml_dhyp = theano.function([self.th_hyp, self.th_X, self.th_Y], self.th_dlml_dhyp)

    def sample_prior(self, hyp, X=None):
        if (X == None):
            if (self.X == None):
                raise ValueError("If no X is passed, it needs to be present in the object.")
            X = self.X

        K = self.kernel.K(X, hyp)
        cK = linalg.cholesky(K)

        # Print smallest eigenvalue
        # print np.min(np.diag(cK)) ** 2.0
        
        z = rnd.randn(X.shape[0])
        return np.dot(cK, z)

    def calc_post(self, hyp, W):
        X = self.X
        Y = self.Y

        Kxx = self.kernel.K(X, hyp)
        Kxw = self.kernel.Kc(X, W, hyp)
        Kww = self.kernel.K(W, hyp)
        m = Kxw.T.dot(linalg.inv(Kxx)).dot(Y)
        cov = Kww - Kxw.T.dot(linalg.inv(Kxx)).dot(Kxw)

        return m, cov

    def sample_post(self, hyp, W):
        m, cov = self.calc_post(hyp, W)

        return linalg.cholesky(cov).dot(rnd.randn(W.shape[0])) + m, m, cov

    def nlml(self, hyp, X=None, Y=None):
        if (X == None):
            X = self.X
        if (Y == None):
            Y = self.Y

        return -self.lml(hyp, X, Y)

    def dnlml_dhyp(self, hyp, X=None, Y=None):
        if (X == None):
            X = self.X
        if (Y == None):
            Y = self.Y

        return -self.dlml_dhyp(hyp, X, Y)
