from math import sqrt

import numpy as np

from bimu.preprocessing.text import n_toks
from bimu.utils.generic_utils import permute

r = np.random.RandomState(1234)


def get_random_w(sampling_tab):
    return r.randint(0, len(sampling_tab)-1)


def build_subsampling_tab(w_cn, inv_w_index, sampling_factor=1e-3):
    """

    :param w_cn: a w : count mapping
    :param inv_w_index: an idx : w mapping
    :param sampling_factor: small for agressive subsampling
    """
    thr = sampling_factor * n_toks(w_cn)
    subsampling_tab = []
    for i in range(1, len(w_cn)+1):
        cn = w_cn[inv_w_index[i]]  # skip 0 (for padding)
        rnd = (sqrt(cn / thr) + 1) * (thr / cn)
        subsampling_tab.append(min(rnd, 1.0))
    assert len(subsampling_tab) == len(w_cn)

    return subsampling_tab


def build_sampling_tab(w_cn, inv_w_index, sampling_tab_size=1e8, power=0.75):
    """
    Precompute table for negative sampling.
    0.75: The more frequent the word, the more downsmoothed, but x^0.75 tends to +inf,
        so the word with a higher count will be sampled with higher p than a less frequent word.

    """
    vocab_size = len(w_cn)

    tab = []
    assert vocab_size != 0

    # Z normalizer: over all pow counts
    z = n_toks(w_cn, lambda x: pow(x, power))

    # fill with idxs proportional to word's cn**power
    idx = 1
    # normalize count^0.75 by Z
    w_prop = w_cn[inv_w_index[idx]]**power / z  # running word proportion
    for tab_idx in range(int(sampling_tab_size)):
        tab.append(idx)
        if tab_idx / sampling_tab_size > w_prop:
            idx += 1
            w_prop += w_cn[inv_w_index[idx]]**power / z
        if idx >= vocab_size:
            idx = vocab_size - 1

    return tab


#v.line_to_seq(line)
def mbatch_skipgrams(reader, v, max_window_size=3, n_neg=1, sampling_tab=None, subsampling_tab=None):
    """
    Avoid checking 0-id contexts and labels in theano training part by building a mask here:

    max_y_is_1: for masking negative and padded contexts/labels. It's an int idx+1 of the last positive context/label;
                we can later compute context mean easily by contexts[:max_y_is_1]

    This datastructure relies on a strict order of instance creation, i.e. positive > negative > padded.
    This is possible because the word order is not important (BOW).

    """
    # minibatch of size 1: pivot word with all context words (including negative samples)
    for n_line, line in enumerate(reader, 1):
        seq = v.line_to_seq(line)
        if len(seq) < 2:
            continue
        for i, w_idx in enumerate(seq):
            if subsampling_tab is not None:
                if subsampling_tab[w_idx-1] < r.random_sample():
                    continue
            eff_window_size = r.randint(1, max_window_size)
            window_start = max(0, i-eff_window_size)
            window_end = min(len(seq), i+eff_window_size+1)
            #window_start = max(0, i-max_window_size)
            #window_end = min(len(seq), i+max_window_size+1)
            contexts = []
            labels = []
            # go over contexts
            for j in range(window_start, window_end):
                if j != i:
                    c_idx = seq[j]
                    # pos
                    contexts.append(c_idx)
                    labels.append(1)
                    # neg
            # to mask irrelevant contexts for context-mean comp. (i.e. negatives and padded):
            max_y_is_1 = len(contexts)
            # negative
            # these strictly follow positives, for ease of masking later
            for _ in range(len(contexts)*n_neg):
                contexts.append(sampling_tab[get_random_w(sampling_tab)])
                labels.append(0)
            assert len(contexts) == len(labels)
            for _ in range(2 * (max_window_size + (max_window_size*n_neg)) - len(contexts)):  # padding for start/end sent & TODO rand window
                contexts.append(0)  # special out of seq idx
                labels.append(0)
            assert len(contexts) == len(labels) == (2 * (max_window_size + (max_window_size*n_neg)))

            yield w_idx, contexts, labels, max_y_is_1, n_line


def mbatch_skipgrams_inference(reader, v, max_window_size=3):
    # minibatch of size 1: pivot word with all context words (including negative samples)
    for n_line, line in enumerate(reader, 1):
        seq = v.line_to_seq(line, output_nan=True)
        #if len(seq) < 2:
        #   continue
        for i, w_idx in enumerate(seq):
            #eff_window_size = r.randint(1, max_window_size)
            eff_window_size = max_window_size
            window_start = max(0, i-eff_window_size)
            window_end = min(len(seq), i+eff_window_size+1)
            #window_start = max(0, i-max_window_size)
            #window_end = min(len(seq), i+max_window_size+1)
            contexts = []
            labels = []
            # go over contexts
            for j in range(window_start, window_end):
                if j != i:
                    c_idx = seq[j]
                    # pos
                    contexts.append(c_idx)
                    labels.append(1)
                    # neg
            # to mask irrelevant contexts for context-mean comp. (i.e. negatives and padded):
            max_y_is_1 = len(contexts)
            assert len(contexts) == len(labels)
            for _ in range(2 * max_window_size - len(contexts)):  # padding for start/end sent & TODO rand window
                contexts.append(0)  # special out of seq idx
                labels.append(0)
            assert len(contexts) == len(labels) == (2 * max_window_size)

            yield w_idx, contexts, labels, max_y_is_1, n_line


def skipgrams(seq, max_window_size=3, n_neg=1, sampling_tab=None, subsampling_tab=None, shuffle=False):
    """
    Return pairs [w_idx, c_idx] and labels (0 or 1), where
        0 means c_idx is randomly sampled and 1 that c_idx
        was observed with w_idx
    """

    pairs = []
    labels = []
    seq_len = len(seq)

    eff_len = 0
    # go over pivots
    for i, w_idx in enumerate(seq):
        #if not w_idx:
        #    continue
        if subsampling_tab is not None:
            if subsampling_tab[w_idx] < r.random_sample():
                continue
        eff_len += 1
        # dynamic window size
        eff_window_size = r.randint(1, max_window_size)
        window_start = max(0, i-eff_window_size)
        window_end = min(seq_len, i+eff_window_size+1)
        # go over contexts
        for j in range(window_start, window_end):
            if j != i:
                c_idx = seq[j]
                # pos
                pairs.append((w_idx, c_idx))
                labels.append(1)
                # neg
                for _ in range(n_neg):
                    pairs.append((w_idx, sampling_tab[get_random_w(sampling_tab)]))
                    labels.append(0)
    assert len(pairs) == len(labels)

    if shuffle:
        seed = r.randint(0, 10e6)
        permute(pairs, seed)
        permute(labels, seed)

    return pairs, labels, eff_len