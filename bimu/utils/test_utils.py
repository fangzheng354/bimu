import numpy as np


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cosines(W, W2):
    if W2.ndim == 2:
        scores = []
        for w_emb in W2:
            scores.append(cosines(W, w_emb))
        return np.array(scores)
    w_emb_norm = np.linalg.norm(W2)
    return np.dot(W, W2) / (np.linalg.norm(W, axis=1) * w_emb_norm)


#def max_cosine(W, W2):
#    if W2.ndim == 2:
#        max_score = 0
#        for w_emb in W2:
#            curr_max = np.max(cosines(W, w_emb))
#            if curr_max > max_score:
#                max_score = curr_max
#        return max_score
#    return np.max(cosines(W, W2))


def avg_sim(W, W2):
    return np.mean(cosines(W, W2))


def max_sim(W, W2):
    return np.max(cosines(W, W2))


def softmax(x):
    assert x.ndim == 1
    return np.exp(x) / np.sum(np.exp(x))


def closest_idxs(w_emb, weights, n_closest):
    if weights.ndim == 3 and w_emb.ndim == 2:
        n_words = weights.shape[0]
        n_senses = weights.shape[1]
        scores = np.empty((n_words, n_senses))  # [i,j] is cosine of maximally similar sense vector from W[i] for the jth sense of the word in question (w_senses_emb[j])
        for i in range(n_words):
            scores[i] = cosines(weights[i], w_emb).max(axis=1)  # n_senses(in W)*n_senses(in w_senses_emb)
        closest = np.argsort(scores, axis=0)[::-1][:n_closest]
        scores = scores[closest.flatten(), list(range(n_senses))*n_closest]
        scores = scores.reshape(closest.shape)
    else:
        assert weights.ndim == 2
        assert w_emb.ndim == 1
        scores = cosines(weights, w_emb)
        closest = np.argsort(scores)[::-1][:n_closest]
        scores = scores[closest]
    return zip(closest, scores)


def embed_w(w, w_index, weights):
    idx = w_index.get(w)
    if idx is None:
        return None
    return weights[idx]


def closest_to_w(w, w_index, inv_w_index, weights, n_closest=10):
    w_emb = embed_w(w, w_index, weights)
    if w_emb is None:
        return []
    if weights.ndim == 3 and w_emb.ndim == 2:
        neighs = []
        for idxs, score in closest_idxs(w_emb, weights, n_closest+1):
            ws = [inv_w_index[idx] for idx in idxs]
            neighs.append(list(zip(ws, score)))
    else:
        assert weights.ndim == 2
        assert w_emb.ndim == 1
        neighs = [[(inv_w_index[idx], score)] for idx, score in closest_idxs(w_emb, weights, n_closest+1)]  # +1 for the target

    return neighs


def closest_to_w_distinct_weights(w, w_index, inv_w_index, weights_emb, weights_comp, n_closest=10):
    """
    Experimental for comparing pivot/context representations
    """
    w_emb = embed_w(w, w_index, weights_emb)
    if w_emb is None:
        return []
    return [(inv_w_index[idx], score) for idx, score in closest_idxs(w_emb, weights_comp, n_closest+1)]  # +1 for the target


# from keras
def closest_to_point(reverse_word_index, norm_weights, point, skip_top=100, nb_closest=10):
    proximities = np.dot(norm_weights, point)
    tups = list(zip(list(range(len(proximities))), proximities))
    tups.sort(key=lambda x: x[1], reverse=True)
    tups = [(t0, t1) for t0, t1 in tups if not t0 < skip_top]
    return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]


# from keras
def closest_to_word(w, word_index, inv_word_index, norm_weights, max_features, skip_top=100, nb_closest=10):
    i = word_index.get(w)
    #if (not i) or (i<skip_top) or (i>=max_features):
    if (not i) or (i>=max_features):
        return []
    return closest_to_point(inv_word_index, norm_weights, norm_weights[i].T, skip_top, nb_closest)


# from keras
def normalize(W, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(W, order, axis))
    l2[l2 == 0] = 1
    return W / np.expand_dims(l2, axis)

