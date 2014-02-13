###############################################################################
# Unit Tests
# Simple tests to confirm the consistancy between these functions here and the
# standard pdf functions for single variables.
###############################################################################
import unittest

import numpy as np
import numpy.random as rnd

import theanogp.kernels as kern

import GPy
import matplotlib.pyplot as plt

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.D = 3
        self.N = 4
        self.X = rnd.randn(self.N, self.D)

        ard = list(rnd.randn(self.D))
        self.hyp = [2.0] + ard + [10**-3]

        self.gpyk = GPy.kern.rbf(self.D, 2.0, lengthscale=ard, ARD=True) + GPy.kern.white(self.D, 10**-3)
        self.k = kern.covSEardJ(self.D)

    def test_self_kern(self):
        gpyK = self.gpyk.K(self.X)
        K = self.k.K(self.X, self.hyp)

        diff = np.sum(np.abs(K - gpyK))
        self.assertTrue(diff < 10**-10)
        
    def test_cross_kern(self):
        Xc = rnd.randn(self.N + 20, self.D)

        gpyK = self.gpyk.K(self.X, Xc)
        K = self.k.Kc(self.X, Xc, self.hyp)

        self.assertTrue(np.all(K.shape == gpyK.shape))

        diff = np.sum(np.abs(K - gpyK))
        self.assertTrue(diff < 10**10)


suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
