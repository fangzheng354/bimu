import sys

from nltk import word_tokenize as tokenizer

with open(sys.argv[1]) as f_in, open(sys.argv[2], "w") as f_out:
    for c, l in enumerate(f_in):
        f_out.write(" ".join(tokenizer(l))+"\n")
        #if c % 1000 == 0:
        #    print(c)