import sys
import time

import numpy as np
import numpy.random as rnd

import scipy.optimize as opt
import scipy.io as sio

import mltools.simple_optimise as mlopt

sys.path.append('..')
import gp as gp
import kernels as kernels

import matplotlib.pyplot as plt

cwdata = sio.loadmat('cw1d.mat')
X = np.array(cwdata['x'])
Y = np.array(cwdata['y'])

# log(sf), log(sl), log(sn)
hyp_se = [0., -1.0, 0.]
# hyp = [1., 0.01, 10**-1]
# hyp = [1., 0.01, -1]

hyp_prod = [1, 1, 1, 0]

k_se = kernels.covSEardJ(1)
d_se = gp.gp(k_se, X, Y)
d_se.sample_prior(hyp_se, X)

k_prod = kernels.covPeriodicJ(1)
d_prod = gp.gp(k_prod, X, Y)
d_prod.sample_prior(hyp_prod, X)

# Print out the likelihood for a few different hyperparameters
print "Start optimisation"

def opt_callback(x, dx=None, f=None):
    # time.sleep(0.5)
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
        if (opt_callback.i % 10) == 0:
            print ''

opt_callback.i = 0

# hyp = mlopt.gradient_descent(d.nlml, hyp, jac=d.dnlml_dhyp, args={'X':X, 'Y':Y}, tol=10**-4, options={'verbosity':1, 'max_eps':0.001, 'momentum':0.1}, callback=opt_callback, maxiter=5000)
optres = opt.minimize(d_se.nlml, hyp_se, jac=d_se.dnlml_dhyp, args=(X, Y), method='BFGS', callback=opt_callback)
print ''
#print optres
hyp_se = optres.x

print hyp_se
print 'lml se  ', d_se.lml_stable(hyp_se, X, Y)

opt_callback.i = 0
optres = opt.minimize(d_prod.nlml, hyp_prod, jac=d_prod.dnlml_dhyp, args=(X, Y), method='BFGS', callback=opt_callback)
print ''
#print optres
hyp_prod = optres.x

print hyp_prod
print 'lml prod', d_prod.lml_stable(hyp_prod, X, Y)

W = np.atleast_2d(np.linspace(-5.0, 5.0, 1000)).T
ps_mean, ps_var = d_se.calc_post(hyp_se, W)
ps_var = np.diag(ps_var)[:, None]
ps_samp, _, _ = d_se.sample_post(hyp_se, W)

plt.plot(X, Y, 'x')
plt.plot(W, ps_mean)
plt.plot(W, ps_mean + 2.0*np.sqrt(ps_var))
plt.plot(W, ps_mean - 2.0*np.sqrt(ps_var))

ps_mean, ps_var = d_prod.calc_post(hyp_prod, W)
ps_var = np.diag(ps_var)[:, None]
ps_samp, _, _ = d_prod.sample_post(hyp_prod, W)

plt.figure()
plt.plot(X, Y, 'x')
plt.plot(W, ps_mean)
plt.plot(W, ps_mean + 2.0*np.sqrt(ps_var))
plt.plot(W, ps_mean - 2.0*np.sqrt(ps_var))
plt.show()
