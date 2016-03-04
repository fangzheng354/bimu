import argparse
from bimu.preprocessing.io import line_reader

from bimu.utils.conll_utils import *
from bimu.utils.generic_utils import load_json

# python3 eval/nn/conll_to_bar.py -input_file ~/Datasets/wsj/wsjtrain.rungsted -output_file ~/Datasets/wsj/wsjtrain.bar -tag_vocab_file ~/Datasets/wsj/wsjtrain.rungsted.tagvocab.json -vocab_file ~/Documents/bilingualmultisense/bimu/eval/wordvectors/sg_czeng.en_lH_lL2_lr0.1_e1e-06_mb1000_min20_max100000_ep3_neg1_s1e08_dim50_del0_downFalse_win5_sfac1e-03_lcrossentropy_oAdagrad/W_v.txt


def read_vocab(vocab_file, vocab_limit):
    if vocab_file.endswith(".json"):
        vocab = load_json(vocab_file)
    else:
        vocab = {l.strip(): c for c, l in enumerate(line_reader(vocab_file))}
    assert vocab["<s>"] == 0
    return {w: i for w, i in vocab.items() if i < vocab_limit}


class ConllToBar():
    def __init__(self, vocab_file, tag_vocab_file, vocab_limit):
        self.vocab = read_vocab(vocab_file, vocab_limit)
        self.tag_vocab = load_json(tag_vocab_file)

    def read_instance(self, input_file):
        reader = Conll07Reader(input_file)
        for sent in reader:
            w_ids, tag_ids = [], []
            for w, tag in zip(sent.form, sent.cpos):
                w_ids.append(self.vocab.get(w, self.vocab["<s>"]))
                tag_ids.append(self.tag_vocab[tag])

            yield w_ids, tag_ids


def process(input_file, output_file, vocab_file, tag_vocab_file, vocab_limit):
    c = ConllToBar(vocab_file, tag_vocab_file, vocab_limit)
    with open(output_file, "w") as out_f:
        for ws, ts in c.read_instance(input_file):
            for w in ws:
                out_f.write("{} ".format(w))
            out_f.write("|")
            for t in ts:
                out_f.write(" {}".format(t))
            out_f.write("\n")
    return output_file, len(c.tag_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", required=True)
    parser.add_argument("-output_file", required=True)
    parser.add_argument("-vocab_file", help="Either in indexed json format or one word/line format",
                        required=True)
    parser.add_argument("-vocab_limit", type=int, default=1000000)
    parser.add_argument("-tag_vocab_file", required=True)
    args = parser.parse_args()

    process(args.input_file, args.output_file, args.vocab_file, args.tag_vocab_file, args.vocab_limit)
