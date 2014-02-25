from scipy import constants as const
from scipy import optimize as opt
import numpy as np
import numpy.linalg as la

import theano as th
import theano.tensor as T
#import theano.sandbox.linalg.ops as sT
from theano.sandbox.linalg import ops as sT

import mltools.prob as mlprob

import time
import os

mu = T.vector('mu')
sigma = T.matrix('sigma')
prec = sT.matrix_inverse(sigma)

x = np.array([[1.5, -1.5], [-1.5, 1.5], [-1.4, 1.5], [1.4, -1.5]])
D = x.shape[1]
N = x.shape[0]
d = x - mu

#p = -(const.pi)T.dot(T.dot(d, sigma), d.T).trace()
#p = T.log((2*const.pi)**(-0.5*D) * )
#p = sT.det(sigma) + T.dot(mu, mu)
p = T.log(T.prod((2*const.pi)**(-0.5*D) * sT.det(sigma)**-0.5 *
     T.exp(sT.diag(-0.5*T.dot(d, T.dot(prec, d.T))))))

dp_dmu = T.grad(p, mu)
dp_dsigma = T.grad(p, sigma)

fp = th.function([mu, sigma], p)
fd = th.function([mu], d)
fdp_dmu = th.function([mu, sigma], dp_dmu)
fdp_dsigma = th.function([mu, sigma], dp_dsigma)

curmu = np.array([7.5, -3.23])
cursig = np.array([[1., 0], [0, 1.]])
# curmu = np.zeros(2)
# cursig = np.dot(x.T, x) / N

# Compare to multivalued normal
print 'mlprob calculation:'
print np.sum(mlprob.mvnlogpdf(x, curmu, cursig))
print 'Theano calculation'
print fp(curmu, cursig)

if not os.path.exists('graphs'):
    os.mkdir('graphs')

th.printing.pydotprint(fp, outfile='./graphs/p.png', var_with_name_simple=True)
th.printing.pydotprint(fdp_dmu, outfile='./graphs/dp_dmu.png', var_with_name_simple=True)
th.printing.pydotprint(fdp_dsigma, outfile='./graphs/dp_dsigma.png', var_with_name_simple=True)

def objfunc(x):
    curmu = x[0:D]
    cursig = np.reshape(x[D:], (D, D))
    return fp(curmu, cursig)

def deriv(x):
    curmu = x[0:D]
    cursig = np.reshape(x[D:], (D, D))

    print curmu
    print cursig

    dmu = fdp_dmu(curmu, cursig)
    dsig = fdp_dsigma(curmu, cursig)

    return np.append(dmu, dsig.flatten())

res = opt.minimize(objfunc, np.append(curmu, cursig.flatten()), method='BFGS', jac=deriv, options={'disp': True})

# sigstep = 0.1

# while (1):
#     curmu += 0.1 * fdp_dmu(curmu, cursig)
#     cursig_prop = cursig + sigstep * fdp_dsigma(curmu, cursig)
#     while (np.min(la.eig(cursig_prop)[0]) < 0):
#         if (sigstep < 10**-5):
#             break
#         sigstep /= 10
#         cursig_prop = cursig + sigstep * fdp_dsigma(curmu, cursig)
#     else:
#         cursig = cursig_prop

#     if (np.random.randint(100) == 1):
#         print 'dmu   ', fdp_dmu(curmu, cursig)
#         print 'logp  ', fp(curmu, cursig)
#         print 'mu    ', curmu
#         print 'sigma ', cursig

#         print ''

# while (1):
#    print (fdp_dmu(a, b), fdp_dsigma(a, b))
#    a += 0.1 * fdp_dmu(a, b)
#    b += 0.1 * fdp_dsigma(a, b)
#    b = max(b, 10**-8)

#    print (a, b)
#    print ''
#    time.sleep(0.01)

