import numpy as np
from bimu.utils.test_utils import cosines, closest_idxs

n_words = 10
n_senses = 3
emb_dim = 5
n_closest = 5
cosine_scores = np.empty((n_words, n_senses))  # [i,j] is cosine of maximally similar sense vector from W[i] for the jth sense of the word in question (w_senses_emb[j])

np.random.seed(5)
w_senses_emb = np.random.rand(n_senses, emb_dim)
W = np.random.rand(n_words, n_senses, emb_dim)
W[2] = w_senses_emb

for i in range(W.shape[0]):
    #print(i)
    #c = cosines(W[i], w_senses_emb)
    #c_max = c.max(axis=1)  # max cosine per each sense in w_senses_emb; n_senses*1
    #cosine_scores[i] = c_max
    cosine_scores[i] = cosines(W[i], w_senses_emb).max(axis=1)  # n_senses(in W)*n_senses(in w_senses_emb)

#print(cosine_scores)
argsorted = np.argsort(cosine_scores, axis=0)[::-1][:n_closest]
scores = cosine_scores[argsorted.flatten(), list(range(n_senses))*argsorted.shape[0]]
scores = scores.reshape(argsorted.shape)
#print(list(zip(argsorted, cosine_scores[argsorted])))

print(list(closest_idxs(w_senses_emb, W, n_closest=5)))