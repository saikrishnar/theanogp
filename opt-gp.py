import sys
import time

import numpy as np
import numpy.random as rnd

import scipy.optimize as opt
import scipy.io as sio

import mltools.simple_optimise as mlopt

import theanogp.gp as gp
import theanogp.kernels as kernels

import matplotlib.pyplot as plt

cwdata = sio.loadmat('cw1d.mat')
X = np.array(cwdata['x'])
Y = np.array(cwdata['y'])

# log(sf2), log(sl), log(sn2)
hyp = [0., -1.0, 0.]
# hyp = [1., 0.01, 10**-1]

k = kernels.covSEardJ(1)
d = gp.gp(k, X, Y)

d.sample_prior(hyp, X)

# Print out the likelihood for a few different hyperparameters
print "Start optimisation"

def opt_callback(x, dx=None, f=None):
#     time.sleep(0.05)
    opt_callback.i += + 1
#     if ((opt_callback.i % 100) == 0):
    if (True):
        sys.stdout.write(str(x) + ' ')
        if (dx != None):
            sys.stdout.write(str(dx) + ' ')
        if (f != None):
            sys.stdout.write(str(f) + ' ')
        sys.stdout.write(' ' + str(opt_callback.i) + ' ')
        sys.stdout.write('\r')
opt_callback.i = 0

# hyp = mlopt.gradient_descent(d.nlml, hyp, jac=d.dnlml_dhyp, args={'X':X, 'Y':Y}, tol=10**-4, options={'verbosity':1, 'max_eps':0.001, 'momentum':0.1}, callback=opt_callback, maxiter=5000)
optres = opt.minimize(d.nlml, hyp, jac=d.dnlml_dhyp, args=(X, Y), method='CG', callback=opt_callback)
print ''
print optres
optres = opt.minimize(d.nlml, optres.x, jac=d.dnlml_dhyp, args=(X, Y), method='CG', callback=opt_callback)
print ''
print optres

hyp = optres.x

W = np.atleast_2d(np.linspace(-5.0, 5.0, 120)).T
ps_mean, ps_var = d.calc_post(hyp, W)
ps_var = np.diag(ps_var)[:, None]
ps_samp, _, _ = d.sample_post(hyp, W)

plt.plot(X, Y, 'x')
plt.plot(W, ps_mean)
plt.plot(W, ps_mean + 2.0*ps_var)
plt.plot(W, ps_mean - 2.0*ps_var)
plt.show()
