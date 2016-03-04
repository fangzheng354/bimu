import random
import numpy as np
import theano
import theano.tensor as T

theano.config.floatX='float32'
E = np.random.randn(6,2).astype('f')
t_E = theano.shared(E)
# t_E.eval()
# index with np.asarray([2, 0]) to get 3rd and 1st row
# index with np.asarray([2, 0], [3,3]) to get two batches: 3rd and 1st row; 4th and 4th row

t_idxs = T.ivector()  # d=2
t_embedding_output = t_E[t_idxs]  # get the two embs
t_dot_product = T.dot(t_embedding_output[0], t_embedding_output[1])

t_label = T.iscalar()  # 0 or 1
cost=abs(t_label - t_dot_product)
gradient = T.grad(cost=cost, wrt=t_E)  #l1 penalty
theano.pp(gradient)
updates = [(t_E, t_E - 0.01 * gradient)]
# sparse updates with inc_subtensor:
#gradient = T.grad(cost=abs(t_label - t_dot_product), wrt=t_embedding_output)
#updates = [(t_E, T.inc_subtensor(t_embedding_output, -0.01 * gradient))]

train = theano.function(inputs=[t_idxs, t_label], outputs=[], updates=updates)


print("i n d0 d1")
for i in range(0, 1000):
    v1, v2 = random.randint(0, 5), random.randint(0, 5)
    label = 1 if (v1/2 == v2/2) else 0
    train([v1, v2], label)

    if i % 100 == 0:
        for n, embedding in enumerate(t_E.get_value()):
            print(i, n, embedding[0], embedding[1])









