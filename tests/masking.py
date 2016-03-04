import numpy as np
import theano
import theano.tensor as TT
from bimu.initializations import uniform
from bimu.models.senses import Senses


def test_means_numpy():
    s = Senses(10,5,3)
    s.build()
    p = np.array([2,5,6])
    X_c = np.array([[3,9,0,0],[4,6,2,3],[5,1,0,0]])
    Y = np.array([[1,0,0,0], [1,1,0,0], [1,0,0,0]])
    max_y = np.array([1,2,1])
    M_pad = X_c > 0
    #s.train(p, X_c, Y, max_y, M_pad)

    # numpy
    c_embs = s.W_c.eval()[X_c.flatten()]  # (n_batches.n_contexts)*emb_dim
    masked_c_embs = c_embs * Y.flatten().reshape(-1, 1)  # all irrelevant to zero
    C_embs = masked_c_embs.reshape(X_c.shape[0], X_c.shape[1], s.emb_dim)  # n_batches*n_contexts*emb_dim
    return np.sum(C_embs, axis=1) / max_y.reshape(-1, 1)  # n_batches*emb_dim


def test_means_numpy_noflatten():
    s = Senses(10,5,3)
    s.build()
    p = np.array([2,5,6])
    X_c = np.array([[3,9,0,0],[4,6,2,3],[5,1,0,0]])
    Y = np.array([[1,0,0,0], [1,1,0,0], [1,0,0,0]])
    max_y = np.array([1,2,1])
    M_pad = X_c > 0
    #s.train(p, X_c, Y, max_y, M_pad)

    # numpy
    c_embs = s.W_c.eval()[X_c]  # (n_batches.n_contexts)*emb_dim
    C_embs = c_embs * Y[:,:,None]  # all irrelevant to zero
    return np.sum(C_embs, axis=1) / max_y.reshape(-1, 1)  # n_batches*emb_dim


def test_means_theano():
    emb_dim = theano.shared(5)
    input_dim = 10
    W_c = uniform((input_dim+1, emb_dim.eval()), name="W_c")

    X_c = TT.imatrix(name="X_c")
    Y = TT.matrix(dtype=theano.config.floatX, name="Y")
    max_y_is_1s = TT.vector(dtype=theano.config.floatX, name="max_y")

    c_embs = W_c[X_c.flatten()]  # (n_batches.n_contexts)*emb_dim
    masked_c_embs = c_embs * Y.flatten().reshape((-1, 1))  # all irrelevant to zero
    C_embs = masked_c_embs.reshape((X_c.shape[0], X_c.shape[1], emb_dim))  # n_batches*n_contexts*emb_dim
    mean = TT.sum(C_embs, axis=1) / max_y_is_1s.reshape((-1, 1))  # n_batches*emb_dim
    f = theano.function(inputs=[X_c, Y, max_y_is_1s],
                    outputs=mean,
                    allow_input_downcast=True,  # ignore mismatch between int and float dtypes
                    mode=None)
    _X_c = np.array([[3,9,0,0],[4,6,2,3],[5,1,0,0]])
    _Y = np.array([[1,0,0,0], [1,1,0,0], [1,0,0,0]])
    _max_y = np.array([1,2,1])
    return f(_X_c, _Y, _max_y)

m_np = test_means_numpy_noflatten()
m_th = test_means_theano()

assert np.allclose(m_np, m_th, atol=1e-4), (m_np, m_th)
print("passed")