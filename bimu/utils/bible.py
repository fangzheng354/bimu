#from nltk import word_tokenize as tokenizer
import argparse

from lxml import etree

"""
Bible: homepages.inf.ed.ac.uk/s0787820/bible/
See also:
-http://link.springer.com/article/10.1007/s10579-014-9287-y
-http://aclweb.org/anthology/P15-2044
"""


def get_text(f, out_f_text, out_f_id):
    def guard(el_text):
        return el_text or ""

    t = etree.iterparse(f, tag="seg")
    with open(out_f_text, "w") as out_text, open(out_f_id, "w") as out_id:
        for _, el in t:
            out_id.write("{}\n".format(el.attrib["id"].strip()))
            out_text.write("{}\n".format(guard(el.text).strip()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", required=True)
    parser.add_argument("-output_file_text")
    parser.add_argument("-output_file_id")
    args = parser.parse_args()

    out_f_text = args.output_file_text or "{}.text".format(args.input_file)
    out_f_id = args.output_file_id or "{}.id".format(args.input_file)

    get_text(args.input_file, out_f_text, out_f_id)
