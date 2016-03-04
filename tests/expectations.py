import numpy as np

from bimu.models.senses_expectation import SensesExpectation


n_batches = 5  # k
n_contexts = 4  # l
dim = 3  # n
n_senses = 2  # j
np.random.seed(5)
W_c_sel = np.random.rand(n_batches, n_contexts, dim)  # n_batches * n_contexts * dim [k,l,n]
np.random.seed(5)
W_sen = np.random.rand(n_batches, n_senses, dim)  # n_batches * n_senses * dim [k,j,n]


def test_numpy():
    dot = np.empty((n_batches, n_contexts))  # [k,l]
    for bat_n in range(W_c_sel.shape[0]):
        Con_emb = W_c_sel[bat_n]  # n_con * dim
        Sen_emb = W_sen[bat_n]  # n_sen * dim
        for con_n in range(Con_emb.shape[0]):
            con = Con_emb[con_n]  # dim
            dot[bat_n, con_n] = np.sum(np.dot(Sen_emb, con.reshape(-1, 1)))  # [1,n]*[n,1] = [1,1]

    return dot


def test_numpy_2():
    """
    remove outer loop of test_numpy():
    = batched_dot
    """
    dot = np.empty((n_batches, n_contexts))  # [k,l]
    for bat_n in range(W_c_sel.shape[0]):
        Con_emb = W_c_sel[bat_n]  # n_con * dim
        Sen_emb = W_sen[bat_n]  # n_sen * dim
        dot[bat_n] = np.sum(np.dot(Con_emb, Sen_emb.transpose()), axis=1)  # [l,1]

    return dot


def test_numpy_3():
    # wrong
    return np.tensordot(W_c_sel, W_sen.transpose(), axes=[(2, 0), (0, 2)])


s = SensesExpectation(10, dim, n_senses)
# s = Senses(10, dim, n_senses)
s.build()

p = np.array([2, 5, 6])  # n_batches=3
X_c = np.array([[3, 9, 0, 0], [4, 6, 2, 3], [5, 1, 0, 0]])  # n_batches*n_contexts
Y = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])  # n_batches*n_contexts
max_y = np.array([1, 2, 1])
M_pad = X_c > 0

print(s.train(p, X_c, Y, max_y, M_pad))
