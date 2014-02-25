import numpy as np

import theano
import theano.tensor as T

class covBase(object):
    '''
    covBase
    Base class for all covariance functions. The following procedures need to be carried out for all kernel functions,
    and are therefore done here:
      - Define the Theano variables for
      - Compile the Theano functions after the derived classes have defined the computation.
    '''
    def __init__(self, D):
        self.D = D

        # Define all the Theano variables
        self.th_hyp = T.vector('hyp')
        self.th_X = T.matrix('X')
        self.th_Xc = T.matrix('Xc')
        self.th_N = self.th_X.shape[0]
        self.th_D = self.th_X.shape[1]

    def _gen_deriv_functions(self):
        '''
        _gen_deriv_functions
        To be called by the derived class to compile all the required functions.
        '''

        ##################################
        # Define some Theano derivatives
        # Derivative w.r.t. the hyperparameters
        self.th_dhyp, uhyp = theano.scan(lambda i, y, x: T.jacobian(y[i], x),
                                         sequences=T.arange(self.th_K.shape[0]),
                                         non_sequences=[self.th_K, self.th_hyp])
        # Derivative w.r.t. the inputs
        self.th_dX, ux = theano.scan(lambda i, y, x: T.jacobian(y[i], x),
                                     sequences=T.arange(self.th_K.shape[0]),
                                     non_sequences=[self.th_K, self.th_X])

        ##################################
        # Compilation
        # Kxx: self covariance matrix
        self.K = theano.function([self.th_X, self.th_hyp], self.th_K)
        # Kxy: cross covariance matrix
        self.Kc = theano.function([self.th_X, self.th_Xc, self.th_hyp], self.th_Kc)
        self.dK_dhyp = theano.function([self.th_X, self.th_hyp], self.th_dhyp, updates=uhyp)
        self.dK_dX = theano.function([self.th_X, self.th_hyp], self.th_dX, updates=ux)

    @property
    def hypD(self):
        raise NotImplementedError

class covSEard(covBase):
    def __init__(self, D):
        super(covSEard, self).__init__(D)

        distmat = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_X) / T.exp(self.th_hyp[1:]))**2,
            2)

        self.th_K = T.exp(self.th_hyp[0]) * T.exp(-distmat / (2.0))

        super(covSEard, self)._gen_deriv_functions()

    @property
    def hypD(self):
        return self.D + 1

class covSEardJ(covBase):
    '''
    covSEardJ
    ARD Squared Exponential covariance function with added jitter.

    Hyperparameters:
     - 0      : sf2 (marginal variance of the GP)
     - 1:(1+D): ard (length scales of each input dimension
     - D+1    : jitter variance
    '''
    def __init__(self, D):
        super(covSEardJ, self).__init__(D)

        distmat = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_X) / T.exp(self.th_hyp[1:-1]))**2,
            2)
        self.th_K = T.exp(self.th_hyp[0]) * T.exp(-distmat / 2.0) + T.eye(self.th_N) * T.exp(self.th_hyp[-1])

        distmat2 = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_Xc) / T.exp(self.th_hyp[1:-1]))**2,
            2)
        self.th_Kc = T.exp(self.th_hyp[0]) * T.exp(-distmat2 / 2.0) + T.eq(distmat2, 0) * T.exp(self.th_hyp[-1])

        super(covSEardJ, self)._gen_deriv_functions()

    @property
    def hypD(self):
        return self.D + 2
