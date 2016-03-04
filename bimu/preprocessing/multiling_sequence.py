from collections import defaultdict
import random

from bimu.preprocessing.sequence import get_random_w


def eff_window(seq, i, max_window_size):
    eff_window_size = random.randint(0, max_window_size)
    window_start = max(0, i-eff_window_size)
    window_end = min(len(seq), i+eff_window_size+1)

    return window_start, window_end


def extract_alignment(line):
    alignment = defaultdict(list)
    for i in line.split():
        i_f, i_e = map(eval, i.split("-"))
        alignment[i_e].append(i_f)
    return alignment


def affiliate(alignment):
    """
    If source aligns to multiple targets, take the middle one and round down when necessary.
    """
    affil = {}
    for i_e, i_f in alignment.items():
        affil[i_e] = i_f[int((len(i_f)-1)/2)]

    return affil


def mbatch_skipgrams_affil(reader_e, reader_f, reader_a, v_e, v_f, max_window_size=3, max_window_size_f=3, n_neg=1,
                           sampling_tab=None, subsampling_tab=None, leaveout_m0=False):
    """
    Avoid checking 0-id contexts and labels in theano training part by building a mask here:

    max_y_is_1: for masking negative and padded contexts/labels. It's an int idx+1 of the last positive context/label;
                we can later compute context mean easily by contexts[:max_y_is_1]

    This datastructure relies on a strict order of instance creation, i.e. positive > negative > padded.
    This is possible because the word order is not important (BOW).

    :param leaveout_m0: whether to leave out the aligned word in l' as part of the l' contexts
    """
    # minibatch of size 1: pivot word with all context words (including negative samples)
    for n_line, (l_e, l_f, l_a) in enumerate(zip(reader_e, reader_f, reader_a), 1):
        s_e = v_e.line_to_seq_pairs(l_e)  # [(i, w_idx), ...]
        s_f = v_f.line_to_seq(l_f, output_nan=True)
        affil = affiliate(extract_alignment(l_a))  # map i_e --> i_f

        if len(s_e) < 2:
            continue
        for i, (orig_i, w_idx) in enumerate(s_e):
            if subsampling_tab is not None:
                if subsampling_tab[w_idx-1] < random.random():
                    continue
            # dynamic window size
            window_start, window_end = eff_window(s_e, i, max_window_size)
            #static window size
            #window_start = max(0, i-max_window_size)
            #window_end = min(len(seq_e), i+max_window_size+1)
            contexts_e = []
            contexts_f = []
            labels = []

            # get contexts_f
            i_f = affil.get(orig_i, None)
            if i_f is not None:
                #c_f_idx = s_f[i_f] or 0  # 0 if OOV
                window_start_f, window_end_f = eff_window(s_f, i_f, max_window_size_f)
                for j in range(window_start_f, window_end_f):
                    if leaveout_m0 and j == i_f:
                        continue
                    contexts_f.append(s_f[j] or 0)
            else:
                # no affiliation
                contexts_f.append(0)
            for _ in range(2*max_window_size_f+1 - len(contexts_f)):
                contexts_f.append(0)

            assert len(contexts_f) == 2*max_window_size_f+1

            # go over contexts_e
            for j in range(window_start, window_end):
                if j != i:
                    c_e_idx = s_e[j][1]
                    # pos
                    contexts_e.append(c_e_idx)
                    labels.append(1)
                    # neg
            # to mask irrelevant contexts for context-mean comp. (i.e. negatives and padded):
            max_y_is_1 = len(contexts_e)
            # negative
            # these strictly follow positives, for ease of masking later
            for _ in range(len(contexts_e)*n_neg):
                contexts_e.append(sampling_tab[get_random_w(sampling_tab)])
                labels.append(0)
            assert len(contexts_e) == len(labels)
            for _ in range(2 * (max_window_size + (max_window_size*n_neg)) - len(contexts_e)):  # padding for start/end sent & TODO rand window
                contexts_e.append(0)  # special out of seq_e idx
                labels.append(0)
            assert len(contexts_e) == len(labels) == (2 * (max_window_size + (max_window_size*n_neg)))

            yield w_idx, contexts_e, contexts_f, labels, max_y_is_1, n_line