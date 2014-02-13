import time

import numpy as np
import numpy.random as rnd

import mltools.simple_optimise as mlopt

import theanogp.gp as gp
import theanogp.kernels as kernels

import matplotlib.pyplot as plt

hyp = [1., 1., 10**-2]
X = np.atleast_2d(np.linspace(0.0, 10.0, 120)).T

kern = kernels.covSEardJ(1)
tgp = gp.gp(kern, X, None)

# Sample something from the prior
pr_samp = tgp.sample_prior(hyp, X)
tgp.Y = pr_samp

# Sample something from the posterior
W = np.atleast_2d(np.linspace(-15.0, 25.0, 300)).T
ps_mean, ps_var = tgp.calc_post(hyp, W)
ps_var = np.diag(ps_var)
ps_samp, _, _ = tgp.sample_post(hyp, W)

plt.plot(X, pr_samp)
plt.plot(W, ps_mean)
plt.plot(W, ps_mean + 2.0*ps_var)
plt.plot(W, ps_mean - 2.0*ps_var)
plt.plot(W, ps_samp)
plt.show()
