import argparse

from bimu.preprocessing.io import line_reader
from bimu.utils.test_utils import *
from bimu.utils.generic_utils import *


def pretty_print(w, neighs):
    def pretty_tup(t):
        """
        :param t: (w, score)
        """
        return "{0} {1}".format(t[0], clrgrn.format(t[1]))

    sep = "\t"
    pad = "{0:40}"
    padmin = "{0:30}"
    clrgrn = "\033[32m{0:.3f}\033[00m"  # green
    if not neighs:
        print(w)
        print("--- NaN")
    else:
        n_sen = len(neighs[0])
        if n_sen == 1:
            print(w)
            print("---")
            for neigh in neighs:
                if neigh[0][0] == w:
                    continue
                print(pretty_tup(neigh[0]))
        else:
            print(sep.join([padmin.format("{}_{}".format(w, s)) for s in range(n_sen)]))
            print(sep.join([padmin.format("{}".format("---")) for _ in range(n_sen)]))
            for neigh in neighs:
                if neigh[0][0] == w:
                    continue
                print(sep.join([pad.format(pretty_tup(t)) for t in neigh]))
    print("\n")

words = [
    "Manchester",
    "rose",
    "apple",
    "Apple",
    "cell",
    "rock",
    "Java",
    "java",
    "space",
    "hit",
    "line",
    "core",
    "max",
    "left",
    "Left",
    "oil",
    "disk",
    "snow",
    "fly",
    "bears",
    "3",
    "two",
    "great",
    "black",
    "star",
    "plant",
    "Indian",
    "France",
    "france",
    "money",
    "years",
    "business",
    "increase"
]


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-input_dir", help="Directory containing model and vocabulary files.", required=True)
parser.add_argument("-ws", default=words, type=str, nargs="+", help="List of words to query.")
parser.add_argument("-ws_file", help="Filepath containing a list of words to query.")
parser.add_argument("-n_closest", type=int, default=15, help="Number of closest words.")
parser.add_argument("-weight_type", default="pivot", choices=["pivot", "context", "both", "shared"],
                    help="Whether to use pivot/context/shared embeddings.")
parser.add_argument("-skip_top", type=int, default=100, help="Number of most frequent words to skip.")
args = parser.parse_args()
print(args.input_dir)

if args.ws_file:
    ws = [w.strip() for w in line_reader(args.ws_file)]
else:
    ws = args.ws

w_index_path = "{}/w_index.json".format(args.input_dir)
w_ind = load_json(w_index_path)
inv_w_ind = {v: k for k, v in w_ind.items()}
print("Loaded vocabulary: {}".format(len(w_ind)))


if args.weight_type == "pivot":
    model_path = "{}/W_w.npy".format(args.input_dir)
    W_w = load_npy(model_path)
elif args.weight_type == "context":
    model_path = "{}/W_c.npy".format(args.input_dir)
    W_c = load_npy(model_path)
elif args.weight_type == "both":
    model_path = "{}/W_w.npy".format(args.input_dir)
    W_w = load_npy(model_path)
    model_path = "{}/W_c.npy".format(args.input_dir)
    W_c = load_npy(model_path)
elif args.weight_type == "shared":
    model_path = "{}/W.npy".format(args.input_dir)
    W = load_npy(model_path)
else:
    sys.exit("Unkown weight type.")
print("Loaded model.")

for word in ws:
    if args.weight_type == "both":
        res = closest_to_w_distinct_weights(word, w_ind, inv_w_ind, W_w, W_c, n_closest=args.n_closest)
    elif args.weight_type == "context":
        res = closest_to_w(word, w_ind, inv_w_ind, W_c, n_closest=args.n_closest)
    elif args.weight_type == "pivot":
        res = closest_to_w(word, w_ind, inv_w_ind, W_w, n_closest=args.n_closest)
    else:
        res = closest_to_w(word, w_ind, inv_w_ind, W, n_closest=args.n_closest)

    pretty_print(word, res)
