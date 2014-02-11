import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg

import theano
import theano.tensor as T

class gp(object):
    def __init__(self, kernel, X, Y):
        self.kernel = kernel
        self.X = X
        self.Y = Y

    def sample(self, hyp):
        K = self.kernel.K(self.X, hyp)
        cK = linalg.cholesky(K)

        # Print smallest eigenvalue
        # print np.min(np.diag(cK)) ** 2.0
        
        z = rnd.randn(self.X.shape[0])
        return np.dot(cK, z)

    def lml(self, hyp):
        pass

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
