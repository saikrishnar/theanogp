import numpy as np

import theano
import theano.tensor as T

class covSEard(object):
    def __init__(self, D, log_sf=1.0, log_ard=None, hyp=None):
        self.D = D
        if (hyp == None):
            if (log_ard == None):
                log_ard = np.ones(D)
            self.hyp = np.hstack(([log_sf, log_ard]))
        else:
            self.hyp = hyp

        hyp = T.vector('hyp')
        X = T.matrix('X')

        distmat = T.sum(
            ((T.reshape(X, (X.shape[0], 1, X.shape[1]) ) - X) / hyp[1:])**2,
            2)

        rbf = hyp[0] * T.exp(-distmat / (2.0))
        drbf_dhyp, uhyp= theano.scan(lambda i, y, x: T.jacobian(y[i], x), sequences=T.arange(rbf.shape[0]), non_sequences=[rbf, hyp])
        drbf_dX, ux = theano.scan(lambda i, y, x: T.jacobian(y[i], x), sequences=T.arange(rbf.shape[0]), non_sequences=[rbf, X])

        # drbf_dsf2, u = theano.scan(lambda i, y, x: T.jacobian(y[i], x), sequences=T.arange(rbf.shape[0]), non_sequences=[rbf, hypsf2])
        # fdrbf_dsf2 = theano.function([X, hypsf2, hypard], drbf_dsf2, updates=u)


        self.K = theano.function([X, hyp], rbf)
        self.dK_dhyp = theano.function([X, hyp], drbf_dhyp, updates=uhyp)
        self.dK_dX = theano.function([X, hyp], drbf_dX, updates=ux)
