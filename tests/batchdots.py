import theano
import theano.tensor as TT
import numpy as np

# define tensor variable
import time

C = TT.matrix("A")  # contexts
T = TT.matrix("B")  # targets
batched_tensor_dot = TT.batched_tensordot(C, T, axes=(0,0))
out = theano.function([C,T],batched_tensor_dot)

# test values
c = np.array([[[1,2,3],[3,4,5]],[[0,1,0],[7,8,6]]])
t = np.array([[1,1,1],[2,2,3]])
start = time.time()
print(out(c, t))
#print(time.time() - start)

# comparison with numpy
print("numpy")
for i in range(t.shape[0]):
    print(np.dot(c[i], t[i]))


