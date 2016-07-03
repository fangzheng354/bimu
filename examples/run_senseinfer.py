"""
Obtain sense predictions for a given tokenized text.
Usage:
python3 examples/run_senseinfer.py -model_load_dir DIRNAME -max_window_size 5 -corpus_path FILENAME -infer_type argmax
This will output the inferred senses to 'FILENAME.senses'.
"""


import argparse
import time

from bimu.models.senses import SensesInference, SensesExpectationInference
from bimu.preprocessing.sequence import *
from bimu.preprocessing.text import *
from bimu.utils.generic_utils import *
from bimu.utils.logger import log
from bimu.utils.test_utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-add_to_dirname", help="String to add to dirname as identifier", type=str)
parser.add_argument("-corpus_path", help="Path to corpus, e.g. data/bllip_10k.", required=True)
parser.add_argument("-max_window_size", help="Maximum window size used in inference.", type=int, default=5)
parser.add_argument("-mbatch_size", help="Minibatch size (n of pivot-contexts pairs).", type=int, default=1000)
parser.add_argument("-infer_type", help="Inference type: argmax or expectation-based.", choices=["argmax", "expect"], required=True)
parser.add_argument("-model_load_dir", help="Path to directory containing a trained model to load.", required=True)
parser.add_argument("-settings_load_fname", help="Filename for settings to load.", default="run_details.json")
parser.add_argument("-settings_save_fname", help="Filename for settings to save.", default="run_details_inference.json")

args = parser.parse_args()

if os.path.isfile(args.settings_load_fname):
    # inherit some settings
    settings = load_json("{}/{}".format(args.model_load_dir, args.settings_load_fname))
else:
    settings = {}
args.n_senses = settings.get("n_senses", 0) or 3
args.emb_dim = settings.get("emb_dim", 0) or 50
args.min_freq = settings.get("min_freq", 0) or 20
args.max_n_words = settings.get("max_n_words", 0) or 1000000
args.discard_n_top_freq = settings.get("discard_n_top_freq", 0) or 0
args.downcase = settings.get("downcase", 0) or False


def save_run_details():
    settings = {
        "corpus_path": args.corpus_path,
        "discard_n_top_freq": args.discard_n_top_freq,
        "downcase": args.downcase,
        "emb_dim": args.emb_dim,
        "max_n_words": args.max_n_words,
        "max_window_size": args.max_window_size,
        "mbatch_size": args.mbatch_size,
        "min_freq": args.min_freq,
        "model_load_dir": args.model_load_dir,
        "n_senses": args.n_senses,
        "settings_save_fname": args.settings_save_fname,
    }
    settings.update({"cwd": os.getcwd()})
    save_json(settings, "{}/{}".format(args.model_load_dir, args.settings_save_fname))


def save_weights(W, str_app=""):
    save_npy(W, "{}/{}{}{}".format(args.model_load_dir, "W_inferred", os.path.basename(args.corpus_path), str_app))


def run(mbatch_size, S_inferred):
    """
    Get counts of senses.
    """
    def update():
        """
        :return: updated counts of senses
        """
        # pivots can repeat, intermediate updates need to be cached
        if modl_type == SensesInference:
            np.add.at(S_inferred, (p, ss), 1)
        elif modl_type == SensesExpectationInference:
            np.add.at(S_inferred, p, ss)
        else:
            sys.exit("Wrong inference type.")


    start = time.time()
    log.info("Starting inference.")
    pivots = []
    contexts = []
    labels = []
    max_y_is_1s = []
    cur_mbatch_size = 0

    for i, (p, cs, ls, max_y_is_1, n_line) in enumerate(mbatch_skipgrams_inference(line_reader(args.corpus_path),
                                                                                      v,
                                                                                      max_window_size=args.max_window_size), 1):
        pivots.append(p)
        contexts.append(cs)
        labels.append(ls)
        max_y_is_1s.append(max_y_is_1)

        cur_mbatch_size += 1
        if cur_mbatch_size != mbatch_size:
            continue
        if pivots:
            x_p = np.array(pivots, dtype="int32")  # n_batches
            X_c = np.array(contexts, dtype="int32")  # n_batches*n_contexts
            L = np.array(labels, dtype="int32")  # n_batches*n_contexts
            max_y = np.array(max_y_is_1s, dtype="int32")  # n_batches
            #M_pad = X_c > 0  # mask for padded
            assert X_c.shape[1] == L.shape[1]
            assert X_c.shape[0] == L.shape[0] == x_p.shape[0] == len(max_y) #== M_pad.shape[0]
            p, ss = modl.train(x_p, X_c, L, max_y)  # n_batches
            update()
        cur_mbatch_size = 0
        pivots = []
        contexts = []
        labels = []
        max_y_is_1s = []

        if i % 10000 == 0:
            rep = "Number of instances: {0}; sequences: {1}".format(i, n_line)
            dyn_print(rep)
    rep = "Number of instances: {0}; sequences: {1}".format(i, n_line)
    dyn_print(rep)
    print("\n")

    log.info("Inference completed: {}s".format(time.time()-start))

    return S_inferred


def run_senseinfer(mbatch_size):
    """
    Obtain sense predictions for every word in the text.
    """
    start = time.time()
    log.info("Starting inference.")
    pivots = []
    contexts = []
    labels = []
    max_y_is_1s = []
    cur_mbatch_size = 0
    all_ss = []

    for i, (p, cs, ls, max_y_is_1, n_line) in enumerate(mbatch_skipgrams_inference(line_reader(args.corpus_path),
                                                                                      v,
                                                                                      max_window_size=args.max_window_size), 1):
        pivots.append(p)
        contexts.append(cs)
        labels.append(ls)
        max_y_is_1s.append(max_y_is_1)

        cur_mbatch_size += 1
        if cur_mbatch_size != mbatch_size:
            continue
        if pivots:
            x_p = np.array(pivots, dtype="int32")  # n_batches
            X_c = np.array(contexts, dtype="int32")  # n_batches*n_contexts
            L = np.array(labels, dtype="int32")  # n_batches*n_contexts
            max_y = np.array(max_y_is_1s, dtype="int32")  # n_batches
            #M_pad = X_c > 0  # mask for padded
            assert X_c.shape[1] == L.shape[1]
            assert X_c.shape[0] == L.shape[0] == x_p.shape[0] == len(max_y) #== M_pad.shape[0]
            p, ss = modl.train(x_p, X_c, L, max_y)  # n_batches
            all_ss.extend(ss)
        cur_mbatch_size = 0
        pivots = []
        contexts = []
        labels = []
        max_y_is_1s = []

        if i % 10000 == 0:
            rep = "Number of instances: {0}; sequences: {1}".format(i, n_line)
            dyn_print(rep)
    # deal with out-of-mbatch
    if pivots:
        x_p = np.array(pivots, dtype="int32")  # n_batches
        X_c = np.array(contexts, dtype="int32")  # n_batches*n_contexts
        L = np.array(labels, dtype="int32")  # n_batches*n_contexts
        max_y = np.array(max_y_is_1s, dtype="int32")  # n_batches
        #M_pad = X_c > 0  # mask for padded
        assert X_c.shape[1] == L.shape[1]
        assert X_c.shape[0] == L.shape[0] == x_p.shape[0] == len(max_y) #== M_pad.shape[0]
        p, ss = modl.train(x_p, X_c, L, max_y)  # n_batches
        all_ss.extend(ss)
    rep = "Number of instances: {0}; sequences: {1}".format(i, n_line)
    dyn_print(rep)
    print("\n")
    log.info("Inference completed: {}s".format(time.time()-start))

    return all_ss


def write_senses(sense_preds):
    """
    Write sense predictions by appending to words,
    like in 'the/1' (here, second sense is assigned, counting from 0).
    """
    corpus_path_senses = "{}.senses".format(args.corpus_path)
    log.info("Writing to {}.".format(corpus_path_senses))
    p_count = 0
    with open(corpus_path_senses, "w") as outfile:
        for l in line_reader(args.corpus_path):
            new_l = []
            for w in l.strip().split(" "):
                new_l.append("{0}/{1}".format(w, sense_preds[p_count]))
                p_count += 1
            outfile.write(" ".join(new_l)+"\n")


log.info("Saving run details to {}.".format("{}/{}".format(args.model_load_dir, args.settings_save_fname)))
save_run_details()

log.info("Building vocabulary.")
v = VocabBuild(min_freq=args.min_freq, max_n_words=args.max_n_words,
               discard_n_top_freq=args.discard_n_top_freq, downcase=args.downcase)
v.load(args.model_load_dir)

#v.save(save_dir)

W_w = load_npy("{}/{}.npy".format(args.model_load_dir, "W_w"))
W_c = load_npy("{}/{}.npy".format(args.model_load_dir, "W_c"))
if args.infer_type == "argmax":
    modl_type = SensesInference
elif args.infer_type == "expect":
    modl_type = SensesExpectationInference
else:
    sys.exit("Wrong inference type.")

modl = modl_type(input_dim=len(v.w_index), emb_dim=args.emb_dim, n_senses=args.n_senses, W_w=W_w, W_c=W_c)

log.info("Compiling the inference model.")
modl.build_inference()

#S_inferred = np.zeros((W_w.shape[0], W_w.shape[1]))  # holds counts of inferred senses
#S_inferred = run(args.mbatch_size, S_inferred)
senses = run_senseinfer(args.mbatch_size)

write_senses(senses)


#log.info("Obtaining weighted-averaged embeddings from inferred senses.")


#def weighted_avg(S, epsilon=1e-6):
#    S_normalized = (S + epsilon) / np.sum(S+epsilon, axis=1).reshape(-1, 1)
#    W_infer = np.mean(W_w*S_normalized[:, :, None], axis=1)

#    return W_infer

#W_inferred = weighted_avg(S_inferred)

#log.info("Saving the inferred embedding matrix.")
#save_weights(W_inferred)