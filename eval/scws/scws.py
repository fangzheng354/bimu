import numpy as np

from bimu.preprocessing.io import line_reader


class Instance():
    def __init__(self):
        self.i = None
        self.w1 = None
        self.pos_w1 = None
        self.w2 = None
        self.pos_w2 = None
        self.w1_sent = None
        self.w2_sent = None
        self.avg_rat = None
        self.ind_rats = None

        self.w1_idx = None
        self.w2_idx = None

    def extract(self, l, w_index, downcase=False):
        l_lst = l.strip().split("\t")
        assert len(l_lst) == 18
        self.i = eval(l_lst[0])
        self.w1 = l_lst[1].lower() if downcase else l_lst[1]
        self.pos_w1 = l_lst[2]
        self.w2 = l_lst[3].lower() if downcase else l_lst[3]
        self.pos_w2 = l_lst[4]
        self.w1_sent = l_lst[5].lower() if downcase else l_lst[5]
        self.w2_sent = l_lst[6].lower() if downcase else l_lst[6]
        self.avg_rat = eval(l_lst[7])
        self.ind_rats = l_lst[8:]

        if w_index is not None:
            self.w1_idx = w_index.get(self.w1, None)
            self.w2_idx = w_index.get(self.w2, None)

    def contexts(self, w_index, win=3):
        """
        Get context ids in a window around target, for both words.
        """
        def contexts_single(sent_inst):
            delim = "<b>"
            sent = sent_inst.split()
            start = sent.index(delim)
            end = start + 3
            left = []
            for w in sent[start-win:start]:
                w_i = w_index.get(w, None)
                if w_i is not None:
                    left.append(w_i)
            right = []
            for w in sent[end:end+win]:
                w_i = w_index.get(w, None)
                if w_i is not None:
                    right.append(w_i)

            return left + right

        return contexts_single(self.w1_sent), contexts_single(self.w2_sent)


class Dataset(list):
    def get_is(self):
        return [item.i for item in self]

    def get_ws1(self):
        return [item.w1 for item in self]

    def get_w1_idxs(self):
        return [item.w1_idx for item in self]

    def get_ws2(self):
        return [item.w2 for item in self]

    def get_w2_idxs(self):
        return [item.w2_idx for item in self]

    def get_pos_ws1(self):
        return [item.pos_w1 for item in self]

    def get_pos_ws2(self):
        return [item.pos_w2 for item in self]

    def get_avg_rats(self):
        return np.array([item.avg_rat for item in self])

    def create(self, f, w_index=None, downcase=False):
        for c, l in enumerate(line_reader(f)):
            inst = Instance()
            inst.extract(l, w_index, downcase=downcase)
            self.append(inst)