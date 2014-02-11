import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg

import scipy.constants as const

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

        self.th_lml = (- 0.5 * sT.trace(T.dot(self.Y.T, T.dot(self.th_K, self.Y))) +
                       - 0.5 * sT.det(self.th_K) +
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

    def nlml(self, **kwargs):
        return -lml(kwargs)

if __name__ == '__main__':
    import kernels
    import matplotlib.pyplot as plt

    # X = np.atleast_2d([0., 1., 2., 3., 3.5, 4.]).T
    Y = np.atleast_2d([0., -1., -2., -3., -3.5, -4.]).T
    X = np.atleast_2d(np.linspace(0.0, 20.0, 120)).T
    hyp = [1., 1., 10**-8]
    
    k = kernels.covSEardJ(1)
    d = gp(k, X, Y)

    s = d.sample(hyp)

    plt.plot(X, s)
    plt.show()
