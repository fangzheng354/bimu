"""
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3 examples/run_bimu.py
"""
import argparse
import time

from bimu.models.bimu import Bimu
from bimu.models.bimu_expectation import BimuExpectation
from bimu.preprocessing.multiling_sequence import mbatch_skipgrams_affil
from bimu.preprocessing.sequence import build_sampling_tab, build_subsampling_tab
from bimu.preprocessing.text import *
from bimu.utils.generic_utils import *
from bimu.utils.logger import log
from bimu.utils.test_utils import *


# np.seterr(all="raise")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-add_to_dirname", help="String to add to dirname as identifier", type=str)
parser.add_argument("-corpus_path_e", help="Path to source lang. corpus.", required=True)
parser.add_argument("-corpus_path_f", help="Path to target lang. corpus.", required=True)
parser.add_argument("-corpus_path_a", help="Path to alignment file.", required=True)
parser.add_argument("-discard_n_top_freq", help="Discard n most frequent words", type=int, default=0)
parser.add_argument("-downcase", help="Downcase words.", action="store_true")
parser.add_argument("-emb_dim", help="Whether to use pivot/context/shared embeddings.", type=int, default=50)
parser.add_argument("-epsilon", help="Numerical stability parameter for optimizer.", type=float, default=1e-6)
parser.add_argument("-lambdaF", help="Weighting factor for the second language context in sense prediction.",
                    type=float, default=0.7)
parser.add_argument("-lambdaH", help="Weighting factor for the entropy regularizer in SensesExpectation.", type=float,
                    default=0.01)
parser.add_argument("-lambdaL2", help="Weighting factor for the L2 regularizer in SensesExpectation.", type=float,
                    default=0.)
parser.add_argument("-leaveout_m0", help="Exclude the affiliated (aligned) word in from the l' context.",
                    action="store_true")
parser.add_argument("-loss_f", help="Loss function to use.", choices=["mse", "crossentropy"], default="crossentropy")
parser.add_argument("-lr", help="Initial learning rate.", type=float, default=0.1)
parser.add_argument("-max_n_words", help="Maximum vocabulary size.", type=int, default=100000)
parser.add_argument("-max_window_size", help="Maximum window size.", type=int, default=5)
parser.add_argument("-max_window_size_f", help="Maximum window size for the foreign language.", type=int, default=0)
parser.add_argument("-mbatch_size", help="Minibatch size (n of pivot-contexts pairs).", type=int, default=1000)
parser.add_argument("-min_freq", help="Minimum frequency of words to take into account.", type=int, default=20)
#parser.add_argument("-min_lr", help="Minimum learning rate for SGDDynamic.", type=float, default=0.001)
parser.add_argument("-model", help="Model to train.", choices=["bimu", "bimu_expect"], required=True)
parser.add_argument("-model_f_dir", help="Directory containing l' model files.", required=True)
#parser.add_argument("-model_load_fname", help="Path to model to load.")
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

args.lambdaH = "" if (args.model != "bimu_expect") else args.lambdaH
args.lambdaL2 = "" if (args.model != "bimu_expect") else args.lambdaL2
save_dir = "output/{}{}{}_{}_lH{}_lL2{}_lF{}_excM0{}_lr{}_e{}_mb{}_min{}_max{}_ep{}_neg{}_s{}_dim{}_del{}_down{}_win{}_winf{}_sfac{}_l{}_o{}".format(
    args.add_to_dirname or "",
    args.model,
    args.n_senses or "",
    args.corpus_path_e.split("/")[-1],
    args.lambdaH,
    args.lambdaL2,
    args.lambdaF,
    args.leaveout_m0,
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
    args.max_window_size_f,
    "1e{}".format("{0:e}".format(args.sampling_fac)[-3:]),
    args.loss_f,
    args.optimizer)


def save_run_details():
    settings = {
        "corpus_path_e": args.corpus_path_e,
        "discard_n_top_freq": args.discard_n_top_freq,
        "downcase": args.downcase,
        "emb_dim": args.emb_dim,
        "epsilon": args.epsilon,
        "lambdaH": args.lambdaH,
        "lambdaL2": args.lambdaL2,
        "lambdaF": args.lambdaF,
        "leaveout_m0": args.leaveout_m0,
        "loss_f": args.loss_f,
        "lr": args.lr,
        "max_n_words": args.max_n_words,
        "max_window_size": args.max_window_size,
        "max_window_size_f": args.max_window_size_f,
        "mbatch_size": args.mbatch_size,
        "min_freq": args.min_freq,
        #"min_lr": args.min_lr,
        "model": args.model,
        "model_f_dir": args.model_f_dir,
        #"model_load": args.model_load,
        #model_load_fname": args.model_load_fname,
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
        contexts_a = []  # affiliated contexts
        labels = []
        max_y_is_1s = []

        for i, (p,
                cs,
                cs_a,
                ls,
                max_y_is_1,
                n_line) in enumerate(mbatch_skipgrams_affil(line_reader(args.corpus_path_e),
                                                            line_reader(args.corpus_path_f),
                                                            line_reader(args.corpus_path_a),
                                                            v_e,
                                                            v_f,
                                                            max_window_size=args.max_window_size,
                                                            max_window_size_f=args.max_window_size_f,
                                                            n_neg=args.n_neg,
                                                            sampling_tab=s_tab,
                                                            subsampling_tab=ss_tab,
                                                            leaveout_m0=args.leaveout_m0), 1):
            if np.all(np.array(ls) == 0):
                continue
            pivots.append(p)
            contexts.append(cs)
            contexts_a.append(cs_a)
            labels.append(ls)
            max_y_is_1s.append(max_y_is_1)

            cur_mbatch_size += 1
            if cur_mbatch_size != mbatch_size:
                continue
            if pivots:
                x_p = np.array(pivots, dtype="int32")  # n_batches
                X_c = np.array(contexts, dtype="int32")  # n_batches*n_contexts
                # l' contexts:
                X_c_f = np.array(contexts_a, dtype="int32")  # n_batches*n_contexts_f
                X_c_f_mask = X_c_f > 0
                L = np.array(labels, dtype="int32")  # n_batches*n_contexts
                max_y = np.array(max_y_is_1s, dtype="int32")  # n_batches
                M_pad = X_c > 0  # mask for padded
                assert X_c.shape[1] == L.shape[1]
                assert X_c.shape[0] == L.shape[0] == x_p.shape[0] == len(max_y) == M_pad.shape[0]
                loss = modl.train(x_p, X_c, X_c_f, X_c_f_mask, L, max_y, M_pad)
                losses.append(loss)

            pivots = []
            contexts = []
            contexts_a = []
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


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

log.info("Saving run details to {}.".format(save_dir))
save_run_details()

log.info("Building vocabulary.")
v_e = VocabBuild(min_freq=args.min_freq, max_n_words=args.max_n_words,
                 discard_n_top_freq=args.discard_n_top_freq, downcase=args.downcase)

r = line_reader(args.corpus_path_e)
v_e.create(r)
log.info("Loading l' vocabulary.")
sett_f = load_json("{}/{}".format(args.model_f_dir, args.settings_save_fname))
v_f = VocabBuild(sett_f["min_freq"],
                 sett_f["max_n_words"],
                 sett_f["discard_n_top_freq"],
                 sett_f["downcase"])
v_f.load(args.model_f_dir)
# for progress report
w1 = v_e.w_index.get("France", v_e.w_index.get("france", ""))
w2 = v_e.w_index.get("Germany", v_e.w_index.get("germany", ""))

v_e.save(save_dir)

s_tab = build_sampling_tab(v_e.w_cn, v_e.inv_w_index, sampling_tab_size=args.sampling_tab_size)
ss_tab = build_subsampling_tab(v_e.w_cn, v_e.inv_w_index, sampling_factor=args.sampling_fac)

model_f_param = load_npy("{}/{}".format(args.model_f_dir, "W_w.npy"))
assert model_f_param.shape[-1] == args.emb_dim

if args.model == "bimu":
    modl = Bimu(input_dim=len(v_e.w_cn), emb_dim=args.emb_dim, n_senses=args.n_senses, W_w_f=model_f_param,
                lambdaF=args.lambdaF)
elif args.model == "bimu_expect":
    # factor for adjusting L2
    adjust = args.mbatch_size / v_e.corpus_size  # corpus size is an approximation
    modl = BimuExpectation(input_dim=len(v_e.w_cn), emb_dim=args.emb_dim, n_senses=args.n_senses, W_w_f=model_f_param,
                           lambdaH=args.lambdaH, lambdaL2=args.lambdaL2, adjust=adjust, lambdaF=args.lambdaF)

log.info("Compiling the model.")
modl.build(loss=args.loss_f, optimizer=args.optimizer, lr=args.lr, epsilon=args.epsilon)

if args.model == "bimu" or args.model == "bimu_expect":
    #sim = avg_sim
    sim = max_sim
else:
    sim = cosine  # for progress report

run(args.n_epochs, args.mbatch_size)

log.info("Saving model.")
save_weights(modl)