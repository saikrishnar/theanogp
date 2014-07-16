import numpy as np

import scipy.constants as const

import theano
import theano.ifelse as theanoifelse
import theano.tensor as T

class covBase(object):
    '''
    covBase
    Base class for all covariance functions. The following procedures need to be carried out for all kernel functions,
    and are therefore done here:
      - Define the Theano variables for
      - Compile the Theano functions after the derived classes have defined the computation.
    '''
    def __init__(self, D, th_X, th_hyp):
        self.D = D

        # Define all the Theano variables
        if th_hyp is None:
            self.th_hyp = T.vector('hyp')
        else:
            self.th_hyp = th_hyp

        if th_X is None:
            self.th_X = T.matrix('X')
        else:
            self.th_X = th_X

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
    def __init__(self, D, th_X=None, th_hyp=None):
        super(covSEard, self).__init__(D, th_X, th_hyp)

        self.th_sf2 = T.exp(self.th_hyp[0] * 2.0)
        self.th_ard = T.exp(self.th_hyp[1:])

        distmat = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_X) / self.th_ard)**2,
            2)

        self.th_K = self.th_sf2 * T.exp(-distmat / (2.0))

        distmat_c_sq = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_Xc) / self.th_ard)**2,
            2)
        self.th_Kc = self.th_sf2 * T.exp(-distmat_c_sq / 2.0)

        super(covSEard, self)._gen_deriv_functions()

    @property
    def hypD(self):
        return self.D + 1

class covSEardJ(covBase):
    '''
    covSEardJ
    ARD Squared Exponential covariance function with added jitter.

    Hyperparameters:
     - 0      : LOG sf (marginal stddev of the GP)
     - 1:(1+D): LOG ard (length scales of each input dimension)
     - D+1    : LOG jitter stddev
    '''
    def __init__(self, D, th_X=None, th_hyp=None, minjitter=10**-8):
        super(covSEardJ, self).__init__(D, th_X, th_hyp)

        self.th_sf2 = T.exp(self.th_hyp[0] * 2.0)
        self.th_ard = T.exp(self.th_hyp[1:-1])
        self.th_sn2 = T.exp(theanoifelse.ifelse(T.lt(2.0 * self.th_hyp[-1], np.log(minjitter)), np.log(minjitter), self.th_hyp[-1] * 2.0))

        distmat_sq = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_X) / self.th_ard)**2,
            2)
        self.th_K = self.th_sf2 * T.exp(-distmat_sq / 2.0) + T.eye(self.th_N) * self.th_sn2

        distmat_c_sq = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_Xc) / self.th_ard)**2,
            2)
        self.th_Kc = self.th_sf2 * T.exp(-distmat_c_sq / 2.0) + T.eq(distmat_c_sq, 0) * self.th_sn2

        super(covSEardJ, self)._gen_deriv_functions()

    @property
    def hypD(self):
        return self.D + 2

class covPeriodicJ(covBase):
    '''
    covPeriodic
    Periodic covariance function with added jitter.

    Hyperparameters:
     - 0: log(sf) (marginal stddev of the GP)
     - 1: log(p) (period)
     - 2: log(l) (length scale of the GP)
     - 3: log(sn) (jitter stddev)
    '''

    def __init__(self, D, th_X=None, th_hyp=None, minjitter=10**-8):
        super(covPeriodicJ, self).__init__(D, th_X, th_hyp)

        self.th_sf2 = T.exp(self.th_hyp[0] * 2.0)
        self.th_p = T.exp(self.th_hyp[1])
        self.th_l2 = T.exp(self.th_hyp[2] * 2.0)
        self.th_sn2 = T.exp(theanoifelse.ifelse(T.lt(2.0 * self.th_hyp[-1], np.log(minjitter)), np.log(minjitter), self.th_hyp[-1] * 2.0))

        distmat = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_X)),
            2)
        self.th_K = self.th_sf2 * T.exp(- 2.0 / self.th_l2 * T.sin(const.pi * distmat / self.th_p)**2) + T.eye(self.th_N) * self.th_sn2

        distmat_c = T.sum(
            ((T.reshape(self.th_X, (self.th_X.shape[0], 1, self.th_X.shape[1]) ) - self.th_Xc)),
            2)
        self.th_Kc = self.th_sf2 * T.exp(- 2.0 / self.th_l2 * T.sin(const.pi * distmat_c / self.th_p)**2) + T.eq(distmat_c, 0) * self.th_sn2

        super(covPeriodicJ, self)._gen_deriv_functions()

    @property
    def hypD(self):
        return 3
