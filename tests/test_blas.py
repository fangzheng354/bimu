import numpy as np
import theano
import theano.tensor as T
import time
rng = np.random

N = 10000
k = T.iscalar("k")
A = T.matrix("A")
D = rng.randn(N, N).astype(theano.config.floatX)
result, updates = theano.scan(fn=lambda prior, A: T.dot(prior, A), non_sequences=A, outputs_info=T.identity_like(A), n_steps=k)

final_result = result[-1]

power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

t0 = time.time()
a = power(D,3)
t1 = time.time()
print(a)
print('Cost time: ', t1 - t0)
