"""
Explore the encoder (sense classifier) predictions.
How sharp are its probabilities, entropy.
"""
import argparse

import numpy as np
from scipy.stats import entropy

from bimu.utils.generic_utils import load_json, load_npy, save_npy, save_model, nparr_to_str
from bimu.utils.logger import log
from bimu.utils.test_utils import softmax
from eval.scws.embed import loc_predict
from eval.scws.scws import Dataset


def get_probs(inst, embs, c_embs, bias, w_index, win_size):
    act1, act2 = loc_predict(inst, embs, c_embs, bias, w_index, win_size)
    return softmax(act1), softmax(act2)


def plot_probs(obj):
    import pylab as pl
    pl.plot(np.sort(obj.flatten()))
    pl.savefig("{}/sense_probs.pdf".format(args.input_dir), bbox_inches='tight')


def get_entropy(obj):
    assert obj.ndim == 2
    e = 0
    for i in obj:
        e += entropy(i)
    return e / obj.shape[0]


def get_entropy_per_type(sense_sel, min_freq = 7):
    """
    Calculate avg entropy by looking at the senses attributed to repeating words.
    So, if a words gets the same sense very often, the entropy would be small.
    """
    frequent = {k: v/v.sum() for k, v in sense_sel.items() if v.sum() > min_freq}  # {word: sense distribution}
    e = 0
    for k, v in frequent.items():
        e += entropy(v)
    return e / len(frequent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", help="Directory containing model and vocabulary files.", required=True)
    parser.add_argument("-f", default="W_w", help="Model file (optional), meant for models per epoch.")
    parser.add_argument("-data_path", help="Filepath containing the SCWS dataset.", default="data/SCWS/ratings.txt")
    parser.add_argument("-win_size", default=3, type=int,
                        help="Context window size (n words to the left and n to the right).")
    parser.add_argument("-n_most_freq", type=int, help="Only consider n most freq. words from vocabulary.")
    args = parser.parse_args()

    w_index_path = "{}/w_index.json".format(args.input_dir)
    # model_path = "{}/sg.pickle".format(args.input_dir)
    log.info("Loading model.")
    w_index = load_json(w_index_path)
    if args.n_most_freq:
        w_index = {w: i for w, i in w_index.items() if i < args.n_most_freq + 1}
        print(len(w_index))

    embs = load_npy("{}/{}.npy".format(args.input_dir, args.f))
    c_embs = load_npy("{}/W_c.npy".format(args.input_dir))
    try:
        if args.f == "W_w":
            n = ""
        else:
            n = eval(args.f[-1])
            assert 0 <= n < 9
        bias = load_npy("{}/Wb{}.npy".format(args.input_dir, n))
    except FileNotFoundError:
        bias = None

    log.info("Loading dataset.")
    d = Dataset()
    d.create(args.data_path, w_index)

    idxs1 = d.get_w1_idxs()
    idxs2 = d.get_w2_idxs()

    log.info("Obtaining probabilities.")
    p_dists = []  # sense probability dists
    d.sense_selected = {}
    for inst in d:
        if inst.w1_idx is None:
            continue
        if inst.w2_idx is None:
            continue
        probs_pair = get_probs(inst, embs, c_embs, bias, w_index, args.win_size)  # 2*n_senses

        # which senses are chosen
        if inst.w1 not in d.sense_selected:
            d.sense_selected[inst.w1] = np.zeros(embs.shape[1], "i")  # n_senses
        if inst.w2 not in d.sense_selected:
            d.sense_selected[inst.w2] = np.zeros(embs.shape[1], "i")  # n_senses
        d.sense_selected[inst.w1][np.argmax(probs_pair[0])] += 1
        d.sense_selected[inst.w2][np.argmax(probs_pair[1])] += 1

        for probs in probs_pair:
            p_dists.append(probs)  # (2 x n_instances)*n_senses

    np_dists = np.array(p_dists)
    log.info("Saving the numpy object.")
    save_npy(np_dists, "{}/sense_probs".format(args.input_dir))
    #log.info("Pickling the sense_selected.")
    #save_model(d.sense_selected, "{}/sense_selected.pkl".format(args.input_dir))
    log.info("Saving flattened.")
    with open("{}/sense_probs.flat".format(args.input_dir), "w") as out_f:
        for i in np.sort(np_dists.flatten()):
            out_f.write("{}\n".format(i))
    log.info("Average entropy of flattened probdist: {}".format(get_entropy(np_dists)))
    log.info("Average entropy of the distribution of selected senses per word type: {}".format(get_entropy_per_type(d.sense_selected, min_freq=0)))

    frequent = {k: v for k, v in d.sense_selected.items() if v.sum() > 7}
    with open("{}/sense_selected.txt".format(args.input_dir), "w") as out_f:
        for k, v in sorted(frequent.items()):
            out_f.write("{0:20} {1}\n".format(k, nparr_to_str(v)))

    log.info("Plot in R: R; source(\"densities.R\") and source(\"dist.R\").")
    #plot_probs(np_dists)
