import numpy as np
import theano.tensor as TT

W_w = np.random.rand(10, 3, 5)
S_inferred = np.arange(30).reshape(10, 3).astype('int')
S_normalized = S_inferred / np.sum(S_inferred, axis=1).reshape(-1, 1)
W_inferred = np.mean(W_w*S_normalized[:, :, None], axis=1)

W_w = np.random.rand(10, 3, 5)
C = np.random.rand(10, 5)
S_inferred = np.arange(30).reshape(10, 3).astype('int')
#S_normalized = S_inferred / np.sum(S_inferred, axis=1).reshape(-1, 1)
S_normalized = TT.nnet.softmax(TT.sum(W_w * C[:, None, :], axis=2))
W_inferred = np.mean(W_w*S_normalized.eval()[:, :, None], axis=1)


