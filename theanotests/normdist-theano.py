from scipy import constants as const
import numpy as np

import theano as th
import theano.tensor as T

import time

mu = T.dscalar('mu')
s2 = T.dscalar('s2')

x = np.array([1.5, -1.5])

p = T.sum( T.log( 1 / T.sqrt(2*const.pi*s2) * T.exp(-(x - mu) ** 2 / (2*s2)) ))

fp = th.function([mu, s2], p)

print fp(0, 1)
dp_dmu = T.grad(p, mu)
dp_ds2 = T.grad(p, s2)
fdp_dmu = th.function([mu, s2], dp_dmu)
fdp_ds2 = th.function([mu, s2], dp_ds2)
print fdp_dmu(1.5, 1)
print fdp_ds2(1.5, 1)

a = -3
b = 10

while (b > 10**-6):
    print (fdp_dmu(a, b), fdp_ds2(a, b))
    a += 0.1 * fdp_dmu(a, b)
    b += 0.1 * fdp_ds2(a, b)
    b = max(b, 10**-8)

    print (a, b)
    print ''
    #time.sleep(0.05)
