import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg

import scipy.constants as const

import time

import theano
import theano.tensor as T
from theano.sandbox.linalg import ops as sT

class gp(object):
    def __init__(self, kernel, X, Y):
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

        self.lml = theano.function([self.th_X, self.th_Y, self.th_hyp], self.th_lml)
        self.dlml_dhyp = theano.function([self.th_X, self.th_Y, self.th_hyp], self.th_dlml_dhyp)

    def sample(self, hyp):
        K = self.kernel.K(self.X, hyp)
        cK = linalg.cholesky(K)

        # Print smallest eigenvalue
        # print np.min(np.diag(cK)) ** 2.0
        
        z = rnd.randn(self.X.shape[0])
        return np.dot(cK, z)

    def sample_post(self, W, X, Y, hyp):
        Kxw = self.kernel.Kc(self.X, W, hyp)
        Kxx = self.kernel.K(X)
        m = Kxw.T.dot(Kxx).dot(Y)

        return linalg.cholesky(Kxw).dot(np.randn(X.shape[0], 1)) + m        

    def nlml(self, **kwargs):
        return -lml(kwargs)

if __name__ == '__main__':
    import kernels
    import matplotlib.pyplot as plt

    # X = np.atleast_2d([0., 1., 2., 3., 3.5, 4.]).T
    # Y = np.atleast_2d([0., -1., -2., -3., -3.5, -4.]).T
    # X = np.atleast_2d(np.linspace(0.0, 20.0, 120)).T
    X = rnd.randn(10, 1)
    hyp = [1., 1., 10**-1]
    
    k = kernels.covSEardJ(1)
    d = gp(k, X, None)

    # Print out the likelihood of a few "typical" samples.
    s = np.atleast_2d(d.sample(hyp)).T
    print d.lml(X, s, hyp)
    s = np.atleast_2d(d.sample(hyp)).T
    print d.lml(X, s, hyp)
    s = np.atleast_2d(d.sample(hyp)).T
    print d.lml(X, s, hyp)
    s = np.atleast_2d(d.sample(hyp)).T
    print d.lml(X, s, hyp)
    s = np.atleast_2d(d.sample(hyp)).T
    print d.lml(X, s, hyp)
    s = np.atleast_2d(d.sample(hyp)).T
    print d.lml(X, s, hyp)
    print ''

    # Print out the likelihood for a few different hyperparameters
    hypOpt = [1., 1., 1.]
    print d.lml(X, s, hypOpt)
    print hypOpt
    print d.dlml_dhyp(X, s, hypOpt)

    print "Start optimisation"
    while (1):
        grad = d.dlml_dhyp(X, s, hypOpt)
        
        hypOpt += grad * 0.001
        print hypOpt, grad, d.lml(X, s, hypOpt)
        if np.sum(np.abs(grad)) < 10**-4:
            break
    
    W = np.atleast_2d(np.linspace(-5.0, 25.0, 120)).T
    plt.plot(W, d.sample_post(W, X, s, hyp))
    plt.show()
