import gzip
import json
import pickle
import sys

import numpy as np
import theano

from bimu.preprocessing.io import line_reader


def get_from_module(identifier, module_params, module_name, instantiate=False):
    """
    Taken from keras.
    """
    if type(identifier) is str:
        res = module_params.get(identifier)
        if not res:
            raise Exception("Invalid {}: {}".format(module_name, identifier))
        if instantiate:
            return res()
        else:
            return res
    return identifier


def save_npy(obj, filename):
    np.save(filename, obj)


def load_npy(filename):
    return np.load(filename)


def save_model(obj, filename):
    with open(filename, 'wb') as out:
        pickle.dump(obj, out)


def load_model(filename):
    with open(filename, 'rb') as in_f:
        return pickle.load(in_f)


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)


def v_to_json(filename, outfilename):
    w_index = {}
    for c, l in enumerate(line_reader(filename)):
        w_index[l.strip()] = c
    save_json(w_index, outfilename)


def dyn_print(data):
    sys.stdout.write("\r\x1b[K{}".format(data))
    sys.stdout.flush()


def nparr_to_str(obj):
    assert len(obj.shape) == 1, "Not a 1-dim. array"
    return " ".join(obj.astype(np.str))


def str_to_nparr(s, sep=" ", typ="f"):
    return np.array(s.strip().split(sep=sep), typ)


def opt_to_mean(o, multi_to_mean=True):
    if o.ndim == 3:
        if multi_to_mean:
            return lambda x: np.mean(x, axis=0)  # avgEmb
        else:
            return lambda x: x.flatten()
    elif o.ndim == 2:
        return lambda x: x  # identity
    else:
        raise NotImplementedError


def _save_w2v(obj, inv_w_index, f):
    """
    Save numpy embeddings as w2v vectors txt file.
    """
    assert obj.shape[0] == len(inv_w_index)
    with open(f, "w") as out_f:
        for i in range(obj.shape[0]):
            out_f.write("{} {}\n".format(inv_w_index[i], nparr_to_str(opt_to_mean(obj)(obj[i]))))


def _save_w2v_sep(obj, inv_w_index, f_e, f_v, multi_to_mean):
    """
    Save numpy embeddings and vocabulary as two txt files.
    """
    assert obj.shape[0] == len(inv_w_index)
    with open(f_e, "w") as out_f_e, open(f_v, "w") as out_f_v:
        for i in range(obj.shape[0]):
            out_f_v.write("{}\n".format(inv_w_index[i]))
            out_f_e.write("{}\n".format(nparr_to_str(
                opt_to_mean(obj, multi_to_mean)(obj[i])
            )))


def save_w2v(dirname, sep_vocab=False, multi_to_mean=True, context_matrix=False):
    if context_matrix:
        np_obj = load_npy("{}/W_c.npy".format(dirname))
        f = "{}/W_c.txt".format(dirname)
        with open(f, "w") as out_f:
            for i in range(np_obj.shape[0]):
                out_f.write("{}\n".format(nparr_to_str(np_obj[i])))
    else:
        np_obj = load_npy("{}/W_w.npy".format(dirname))
        inv_w_index = {i: w for w, i in (load_json("{}/w_index.json".format(dirname))).items()}
        if sep_vocab:  # write to separate vocab and embeddings files
            f_e = "{}/W_e.txt".format(dirname)
            f_v = "{}/W_v.txt".format(dirname)
            _save_w2v_sep(np_obj, inv_w_index, f_e, f_v, multi_to_mean)
        else:
            f = "{}/W_w.txt".format(dirname)
            _save_w2v(np_obj, inv_w_index, f)


def save_w2v_to_sep(dirname):
    with open("{}/W_e.txt".format(dirname), "w") as out_f_e, open("{}/W_v.txt".format(dirname), "w") as out_f_v:
        for l in line_reader("{}/W_w.txt".format(dirname)):
            w, e = l.split(" ", 1)
            out_f_v.write("{}\n".format(w))
            out_f_e.write(e)


def load_w2v(f):
    """
    Loads word2vec-format embeddings.
    """
    ws = []
    with open(f) as in_f:
        m, n = map(eval, in_f.readline().strip().split())
    e_m = np.zeros((m, n))
    for c, l in enumerate(line_reader(f, skip=1)):  # skip dimensions
        w, *e = l.strip().split()
        assert len(e) == n
        if not w or not e:
            print("Empty w or e.")
        ws.append(w)
        e_m[c] = e
    assert len(ws) == e_m.shape[0]
    w_index = {w: c for c, w in enumerate(ws)}

    return w_index, e_m


#def load_mssg(f):
#    """
#    Loads Neelakantan et al.-style embeddings, gz-compressed
#    """
#    with gzip.open(f, "rt") as in_f:
#        #m, n = map(eval, in_f.readline().strip().split())
#        v_size, dim, n_senses, _ = map(eval, in_f.readline().strip().split())
#
#        w_index = {"<s>": 0}
#        embs = np.zeros((v_size, n_senses, dim))
#
#        while True:
#            l = in_f.readline().strip().split()
#            if not l:
#                break
#            if len(l) == 1:
#                w = ""
#                s = l[-1]
#            else:
#                w, s = l
#            assert eval(s) == n_senses
#            i = len(w_index)
#            w_index[w] = i
#            _ = in_f.readline()  # global vector
#            for s_i in range(eval(s)):
#                e = str_to_nparr(in_f.readline())
#                assert e.shape[0] == dim
#                embs[i, s_i] = e
#                _ = in_f.readline()  # cluster mean vector
#    assert len(w_index) == embs.shape[0]
#    return w_index, embs


def permute(l, seed):
    np.random.seed(seed)
    np.random.shuffle(l)


def detect_nan(i, node, fn):
    """
    NB: Try first with np.seterr(all="raise") and with FAST_COMPILE theano flag.

    If this doesn't work, try this, which is from http://deeplearning.net/software/theano/tutorial/debug_faq.html.
    """
    for output in fn.outputs:
        if not isinstance(output[0], np.random.RandomState) and np.isnan(output[0]).any():
            print('*** NaN detected ***')
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            break

if __name__ == "__main__":
    #v_to_json(sys.argv[1], sys.argv[2])
    #save_w2v(sys.argv[1], context_matrix=True)
    context_matrix = False
    if len(sys.argv) == 3:
        if sys.argv[2] == "W_c.npy":
            context_matrix = True

    save_w2v(sys.argv[1], sep_vocab=True, multi_to_mean=False, context_matrix=context_matrix)
    #save_w2v_to_sep(sys.argv[1])