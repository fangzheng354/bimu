import argparse
from bimu.utils.conll_utils import Conll07Reader
from bimu.utils.generic_utils import save_json, load_json


class TagVocab(dict):
    def update_tags(self, input_file):
        reader = Conll07Reader(input_file)
        for sent in reader:
            for tag in sent.cpos:
                if tag not in self:
                    self[tag] = len(self)

    def write(self, output_file):
        save_json(self, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", required=True)
    parser.add_argument("-append", action="store_true")
    parser.add_argument("-output_file", required=True)
    args = parser.parse_args()

    tag_vocab = TagVocab()
    if args.append:
        tag_vocab.update(load_json(args.output_file))
    tag_vocab.update_tags(args.input_file)
    tag_vocab.write(args.output_file)
