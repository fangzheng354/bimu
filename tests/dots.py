import theano
import theano.tensor as T
import numpy as np

# define tensor variable
import time

A = T.matrix("A")
B = T.matrix("B")

results, updates = theano.scan(lambda A_i, B_i: T.dot(A_i,B_i), sequences=[A,B])
compute_norm_cols = theano.function(inputs=[A,B], outputs=[results])

# test value
#a = np.array([[1, 2, 3, 4], [2, 2, 2, 2]], dtype=theano.config.floatX)
#b = np.array([[2, 3, 4, 5], [3, 3, 3, 3]], dtype=theano.config.floatX)
a = np.random.rand(100000, 200).astype(theano.config.floatX)
b = np.random.rand(100000, 200).astype(theano.config.floatX)

start = time.time()
for _ in range(10):
    compute_norm_cols(a,b)[0]
print(time.time() - start)
# comparison with numpy
print(np.sum(a*b,axis=1))

