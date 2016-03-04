import os

from bimu.utils.generic_utils import save_json, load_json
from bimu.utils.logger import log


def down(_, downcase=False):
        return _.lower() if downcase else _


def update_counts(reader, w_cn=None, downcase=False, sep=" "):
    if w_cn is None:
        w_cn = {}
    for l in reader:
        for w in l.strip().split(sep):
            w = down(w, downcase)
            if w not in w_cn:
                w_cn[w] = 0
            w_cn[w] += 1
    return w_cn


def sort_vocab(w_cn):
    return sorted(w_cn.items(), key=lambda x: x[1], reverse=True)


def prune_freq(w_cn_list, min_freq):
    return [i for i in w_cn_list if i[1] >= min_freq]


def n_toks(w_cn, fun=lambda x: x):
    return sum(fun(cn) for cn in w_cn.values())


class VocabBuild():
    def __init__(self, min_freq=10, max_n_words=None, discard_n_top_freq=0, downcase=False, sep=" "):
        self.min_freq = min_freq
        self.max_n_words = max_n_words
        self.discard_n_top_freq = discard_n_top_freq
        self.downcase = downcase
        self.sep = sep

        self.w_cn = {}  # word counts
        self.w_index = {}
        self.inv_w_index = {}

    def create(self, reader):
        w_cn_lst = sort_vocab(update_counts(reader, downcase=self.downcase, sep=self.sep))
        log.debug("Vocabulary size after sorting: {}.".format(len(w_cn_lst)))
        w_cn_lst = w_cn_lst[self.discard_n_top_freq:self.max_n_words]

        if self.min_freq > 1:
            w_cn_lst = prune_freq(w_cn_lst, self.min_freq)

        self.corpus_size = sum(i[1] for i in w_cn_lst)

        # create index
        self.w_index["<s>"] = 0  # for padding
        self.inv_w_index[0] = "<s>"
        for idx, (w, _) in enumerate(w_cn_lst, 1):
            self.w_index[w] = idx
            self.inv_w_index[idx] = w

        self.w_cn = dict(w_cn_lst)
        log.debug("Vocabulary size: {}.".format(len(self.w_index)))

    def line_to_seq(self, line, output_nan=False):
        seq = []
        for w in line.strip().split(self.sep):
            if w in self.w_index:
                seq.append(self.w_index[down(w, self.downcase)])
            else:
                if output_nan:
                    seq.append(None)
                else:
                    continue

        return seq

    def line_to_seq_pairs(self, line):
        """
        Outputs a list of (index, w_idx) tuples.
        """
        seq = []
        for c, w in enumerate(line.strip().split(self.sep)):
            if w in self.w_index:
                seq.append((c, self.w_index[down(w, self.downcase)]))

        return seq

    def save(self, save_dir):
        log.info("Saving vocabulary.")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_json(self.w_index, "{}/w_index.json".format(save_dir))
        save_json(self.w_cn, "{}/w_cn.json".format(save_dir))
        #save_json(self.inv_w_index, "{}/inv_w_index.json".format(save_dir))

    def load(self, load_dir):
        self.w_index = load_json("{}/w_index.json".format(load_dir))
        self.inv_w_index = {i: w for w, i in self.w_index.items()}
        if os.path.isfile("{}/w_cn.json".format(load_dir)):
            self.w_cn = load_json("{}/w_cn.json".format(load_dir))

