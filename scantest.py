import theano
import theano.tensor as T

# Vector function, vector input
x = T.dvector('x')
y = T.vector('y')
z = x ** 2 + y*x
Jx, updatesx = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(z.shape[0]), non_sequences=[z,x])
Jy, updatesy = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(z.shape[0]), non_sequences=[z,y])
fx = theano.function([x, y], Jx, updates=updatesx)
fy = theano.function([x, y], Jy, updates=updatesy)
print fx([4, 4], [1, 1])
print fy([4, 4], [1, 1])

# Matrix function, matrix input
X = T.matrix('X')
Y = T.matrix('Y')
Z = X ** 2 + Y
JX, updatesX = theano.scan(lambda i, y, x: T.jacobian(y[i], x), sequences=T.arange(Z.shape[0]), non_sequences=[Z, X])
fX = theano.function([X, Y], JX, updates=updatesX)
res = fX([[1, 2], [3, 4]], [[0, 0], [0, 0]])
print res
print res.shape
print type(res)

# Scalar function, matrix input
p = X
Jp = T.jacobian(p, X)
fp = theano.function([X], Jp)
print fp([[1, 2], [3, 4]])
