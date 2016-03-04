"""
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3 examples/run_mono.py
"""
import argparse
import time

from bimu.models.skipgram import Skipgram
from bimu.models.senses import Senses
from bimu.models.senses_expectation import SensesExpectation
from bimu.preprocessing.sequence import *
from bimu.preprocessing.text import *
from bimu.utils.generic_utils import *
from bimu.utils.logger import log
from bimu.utils.test_utils import *

# np.seterr(all="raise")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-add_to_dirname", help="String to add to dirname as identifier", type=str)
parser.add_argument("-corpus_path", help="Path to corpus, e.g. data/bllip_10k.", required=True)
parser.add_argument("-discard_n_top_freq", help="Discard n most frequent words", type=int, default=0)
parser.add_argument("-downcase", help="Downcase words.", action="store_true")
parser.add_argument("-emb_dim", help="Whether to use pivot/context/shared embeddings.", type=int, default=50)
parser.add_argument("-epsilon", help="Numerical stability parameter for optimizer.", type=float, default=1e-6)
parser.add_argument("-lambdaH", help="Weighting factor for the entropy regularizer in SensesExpectation.", type=float,
                    default=1.)
parser.add_argument("-lambdaL2", help="Weighting factor for the L2 regularizer in SensesExpectation.", type=float,
                    default=1.)
parser.add_argument("-loss_f", help="Loss function to use.", choices=["mse", "crossentropy"], default="crossentropy")
parser.add_argument("-lr", help="Initial learning rate.", type=float, default=0.1)
parser.add_argument("-max_n_words", help="Maximum vocabulary size.", type=int, default=100000)
parser.add_argument("-max_window_size", help="Maximum window size.", type=int, default=5)
parser.add_argument("-mbatch_size", help="Minibatch size (n of pivot-contexts pairs).", type=int, default=1000)
parser.add_argument("-min_freq", help="Minimum frequency of words to take into account.", type=int, default=20)
#parser.add_argument("-min_lr", help="Minimum learning rate for SGDDynamic optimizer.", type=float, default=0.001)
parser.add_argument("-model", help="Model to train.", choices=["sg", "senses", "senses_expect"], required=True)
parser.add_argument("-model_load_dir", help="Path to directory containing a trained model to load.")
#parser.add_argument("-model_save_fname", help="Filename of model to save.", default="sg.pickle")
parser.add_argument("-n_epochs", help="Number of epochs.", type=int, default=3)
parser.add_argument("-n_neg", help="Number of random words for negative sampling.", type=int, default=1)
parser.add_argument("-n_senses", help="Number of senses for the 'senses' model.", type=int, default=3)
parser.add_argument("-optimizer", help="Optimization technique.",
                    choices=["sgd", "rmsprop", "Adagrad"], default="Adagrad")
#parser.add_argument("-rho", help="Decay parameter for RMSprop", type=float, default=0.9)
parser.add_argument("-sampling_fac", help="Sampling factor for subsampling of frequent words.",
                    type=float, default=1e-3)
parser.add_argument("-sampling_tab_size", help="Size of the table for sampling of negative words.",
                    type=float, default=1e8)
parser.add_argument("-save_model_per_ep", help="Save model after each epoch.", action="store_true")
parser.add_argument("-settings_save_fname", help="Filename for settings to save.", default="run_details.json")

args = parser.parse_args()

if args.model_load_dir is not None:
    settings = load_json("{}/{}".format(args.model_load_dir, args.settings_save_fname))
    # inherit some settings
    args.emb_dim = settings["emb_dim"]
    args.min_freq = settings["min_freq"]
    args.max_n_words = settings["max_n_words"]
    args.discard_n_top_freq = settings["discard_n_top_freq"]
    args.downcase = settings["downcase"]

args.n_senses = args.n_senses if (args.model == "senses" or args.model == "senses_expect") else 0
args.lambdaH = "" if (args.model != "senses_expect") else args.lambdaH
args.lambdaL2 = "" if (args.model != "senses_expect") else args.lambdaL2
save_dir = "output/{}{}{}_{}_lH{}_lL2{}_lr{}_e{}_mb{}_min{}_max{}_ep{}_neg{}_s{}_dim{}_del{}_down{}_win{}_sfac{}_l{}_o{}{}".format(
    args.add_to_dirname or "",
    args.model,
    args.n_senses or "",
    args.corpus_path.split("/")[-1],
    args.lambdaH,
    args.lambdaL2,
    args.lr,
    "1e{}".format("{0:e}".format(args.epsilon)[-3:]),
    args.mbatch_size,
    args.min_freq,
    args.max_n_words,
    args.n_epochs,
    args.n_neg,
    "1e{}".format("{0:e}".format(args.sampling_tab_size)[-2:]),
    args.emb_dim,
    args.discard_n_top_freq,
    args.downcase,
    args.max_window_size,
    "1e{}".format("{0:e}".format(args.sampling_fac)[-3:]),
    args.loss_f,
    args.optimizer,
    "_pre" if args.model_load_dir else "")  # pretrained


def save_run_details():
    settings = {
        "corpus_path": args.corpus_path,
        "discard_n_top_freq": args.discard_n_top_freq,
        "downcase": args.downcase,
        "emb_dim": args.emb_dim,
        "epsilon": args.epsilon,
        "lambdaH": args.lambdaH,
        "lambdaL2": args.lambdaL2,
        "loss_f": args.loss_f,
        "lr": args.lr,
        "max_n_words": args.max_n_words,
        "max_window_size": args.max_window_size,
        "mbatch_size": args.mbatch_size,
        "min_freq": args.min_freq,
        #"min_lr": args.min_lr,
        "model": args.model,
        "model_load_dir": args.model_load_dir,
        #"model_save_fname": args.model_save_fname,
        "n_epochs": args.n_epochs,
        "n_neg": args.n_neg,
        "n_senses": args.n_senses,
        "optimizer": args.optimizer,
        #"rho": args.rho,
        "sampling_fac": args.sampling_fac,
        "sampling_tab_size": args.sampling_tab_size,
        "save_dir": save_dir,
        "save_model_per_ep": args.save_model_per_ep,
        "settings_save_fname": args.settings_save_fname,
    }
    settings.update({"cwd": os.getcwd()})
    save_json(settings, "{}/{}".format(save_dir, args.settings_save_fname))


def save_weights(m, str_app="", save_ctx=True):
    if hasattr(m, "W"):
        save_npy(m.W.get_value(), "{}/{}{}".format(save_dir, "W", str_app))
    else:
        save_npy(m.W_w.get_value(), "{}/{}{}".format(save_dir, "W_w", str_app))
        if save_ctx:
            save_npy(m.W_c.get_value(), "{}/{}{}".format(save_dir, "W_c", str_app))
        if hasattr(m, "Wb"):
            save_npy(m.Wb.get_value(), "{}/{}{}".format(save_dir, "Wb", str_app))


def run(nepochs, mbatch_size):
    start = time.time()
    for e in range(nepochs):
        log.info("Epoch {}.".format(e))
        losses = []
        cur_mbatch_size = 0
        print_size = 0

        pivots = []
        contexts = []
        labels = []
        max_y_is_1s = []

        for i, (p, cs, ls, max_y_is_1, n_line) in enumerate(mbatch_skipgrams(line_reader(args.corpus_path),
                                                                             v,
                                                                             max_window_size=args.max_window_size,
                                                                             n_neg=args.n_neg,
                                                                             sampling_tab=s_tab,
                                                                             subsampling_tab=ss_tab), 1):
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
                M_pad = X_c > 0  # mask for padded
                assert X_c.shape[1] == L.shape[1]
                assert X_c.shape[0] == L.shape[0] == x_p.shape[0] == len(max_y) == M_pad.shape[0]
                if args.model == "senses" or args.model == "senses_expect":
                    loss = modl.train(x_p, X_c, L, max_y, M_pad)
                else:
                    loss = modl.train(x_p, X_c, L)
                losses.append(loss)

            pivots = []
            contexts = []
            labels = []
            max_y_is_1s = []

            print_size += cur_mbatch_size
            if print_size % 10000 == 0:
                if not w1 or not w2:
                    sc = "-"
                else:
                    sc = sim(modl.params[0].get_value()[w1], modl.params[0].get_value()[w2])
                rep = "Number of instances: {0}; sequences: {1}; Fr/Ge pivot cos:{2}".format(print_size,
                                                                                             n_line,
                                                                                             sc)
                if len(losses) > 0:
                    rep = "{} {}".format(rep, "Mean loss {}.".format(np.mean(losses)))
                    losses = []
                dyn_print(rep)
            cur_mbatch_size = 0
        print("\n")
        if args.save_model_per_ep and not e == nepochs - 1:  # save of last epoch done below
            save_weights(modl, str_app=str(e))
    log.info("Training completed: {}s/epoch".format((time.time() - start) / nepochs))


def run_fastskipgram(nepochs, mbatch_size):
    start = time.time()
    total_eff_len = 0
    for e in range(nepochs):
        log.info("Epoch {}.".format(e))
        losses = []
        n_instances = 0
        cur_mbatch_size = 0

        pairs = []
        labels = []

        for i, line in enumerate(line_reader(args.corpus_path), 1):
            ps, ls, eff_len = skipgrams(v.line_to_seq(line),
                                        max_window_size=args.max_window_size,
                                        n_neg=args.n_neg,
                                        sampling_tab=s_tab,
                                        subsampling_tab=ss_tab)
            pairs.extend(ps)
            labels.extend(ls)
            cur_mbatch_size += 1
            total_eff_len += eff_len
            if cur_mbatch_size != mbatch_size:
                continue
            else:
                cur_mbatch_size = 0
            #if args.optimizer == "SGDDynamic":
            #    lrate = max(args.min_lr, args.lr * (1 - total_eff_len / total_len))
            #else:
            lrate = args.lr
            if pairs:
                X = np.array(pairs, dtype="int32")
                labs = np.array(labels).reshape(-1, 1)
                #loss = sg.train(X, labs, lrate)
                loss = modl.train(X, labs)
                losses.append(loss)
                n_instances += len(labels)

            pairs = []
            labels = []

            if i % 10000 == 0:
                if not w1 or not w2:
                    sc = "-"
                else:
                    sc = sim(modl.params[0].get_value()[w1], modl.params[0].get_value()[w2])
                rep = "Number of instances: {0}; sequences: {1}; Fr/Ge pivot cos:{2}".format(n_instances,
                                                                                             i,
                                                                                             sc)
                if len(losses) > 0:
                    rep = "{} {}".format(rep, "Mean loss over last {} seqs: {}.".format(len(losses) * mbatch_size,
                                                                                        np.mean(losses)))
                    losses = []
                dyn_print(rep)
        print("\n")
    log.info("Training completed: {}s/epoch".format((time.time() - start) / nepochs))


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

log.info("Saving run details to {}.".format(save_dir))
save_run_details()

log.info("Building vocabulary.")
v = VocabBuild(min_freq=args.min_freq, max_n_words=args.max_n_words,
               discard_n_top_freq=args.discard_n_top_freq, downcase=args.downcase)
if args.model_load_dir is not None:
    v.load(args.model_load_dir)
else:
    r = line_reader(args.corpus_path)
    v.create(r)
# for progress report
w1 = v.w_index.get("France", v.w_index.get("france", ""))
w2 = v.w_index.get("Germany", v.w_index.get("germany", ""))

v.save(save_dir)

s_tab = build_sampling_tab(v.w_cn, v.inv_w_index, sampling_tab_size=args.sampling_tab_size)
ss_tab = build_subsampling_tab(v.w_cn, v.inv_w_index, sampling_factor=args.sampling_fac)

if args.model == "senses_expect":
    if args.model_load_dir:
        raise NotImplementedError
    # factor for adjusting L2
    adjust = args.mbatch_size / v.corpus_size  # corpus size is an approximation
    modl = SensesExpectation(input_dim=len(v.w_cn), emb_dim=args.emb_dim, n_senses=args.n_senses, lambdaH=args.lambdaH,
                             lambdaL2=args.lambdaL2, adjust=adjust)
elif args.model == "senses":
    W_w = load_npy("{}/{}.npy".format(args.model_load_dir, "W_w")) if args.model_load_dir else None
    W_c = load_npy("{}/{}.npy".format(args.model_load_dir, "W_c")) if args.model_load_dir else None
    modl = Senses(input_dim=len(v.w_cn), emb_dim=args.emb_dim, n_senses=args.n_senses, W_w=W_w, W_c=W_c)
else:
    if args.model_load_dir:
        raise NotImplementedError
    modl = Skipgram(input_dim=len(v.w_cn), emb_dim=args.emb_dim)

log.info("Compiling the model.")
modl.build(loss=args.loss_f, optimizer=args.optimizer, lr=args.lr, epsilon=args.epsilon)

if args.model == "senses" or args.model == "senses_expect":
    #sim = avg_sim
    sim = max_sim
else:
    sim = cosine  # for progress report

run(args.n_epochs, args.mbatch_size)

log.info("Saving model.")
save_weights(modl)