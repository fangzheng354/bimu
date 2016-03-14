__author__ = "Jiwei Li, Simon Šuster"
"""
Usage: python3 eval/nn/score.py --help

This is a slightly modified version of Jiwei Li's implementation of a NN POS/NE tagger.
"""

import argparse
import time

from numpy import *
from sklearn.metrics import f1_score
import theano
import theano.tensor as T

from bimu.utils.generic_utils import load_npy
from bimu.utils.logger import log
from eval.nn.conll_to_bar import process


def softmax(x):
    assert x.ndim == 1
    return exp(x) / sum(exp(x))


class ScoreEmbedding:
    def __init__(self, embedding_file, vocab_file, oov=-1, hidden1=50, hidden2=50, hidden3=50, win=7, alpha=0.025):
        self.hidden1, self.hidden2, self.hidden3 = hidden1, hidden2, hidden3
        self.win = win
        self.alpha = alpha
        self.oov = oov
        self.embedding_matrix = self.load_embeddings(embedding_file)
        self.dimension = self.embedding_matrix.shape[-1]
        log.debug("Embedding matrix n rows: {}".format(self.embedding_matrix.shape[0]))
        log.debug("Embedding dimension: {}".format(self.dimension))

    def load_embeddings(self, embedding_file):
        embedding_matrix = load_npy(embedding_file)
        if embedding_matrix.ndim == 2:
            return embedding_matrix
        else:
            assert embedding_matrix.ndim == 3
            # avgEmb (uniform mean)
            return embedding_matrix.mean(axis=1)

    def contextwin(self, l):
        lpad = self.win // 2 * [self.oov] + l + self.win // 2 * [self.oov]
        for i in range(len(l)):
            vector = self.embedding_matrix[lpad[i:(i + self.win)]].reshape(1, self.win * self.dimension)
            if i == 0:
                x = vector
            else:
                x = concatenate((x, vector), axis=0)
        return x  # len(l) by win*dimension

    def Sequence_Level(self, train_file, test_file, num_label, epochs):
        log.debug("Declaring theano vars.")
        random.seed(5)
        W1 = theano.shared(0.2 * random.random([self.win * self.dimension, self.hidden1]) - 0.1)
        W2 = theano.shared(0.2 * random.random([self.hidden1, self.hidden2]) - 0.1)
        W3 = theano.shared(0.2 * random.random([self.hidden2, self.hidden3]) - 0.1)
        U = theano.shared(0.2 * random.random([self.hidden3, num_label]) - 0.1)

        x = T.dmatrix("x")  # len(l) by win*dimension
        y = T.lvector("y")
        learn_rate = T.scalar("learn_rate")

        A1 = T.dot(x, W1)
        B1 = A1 * (A1 > 0)
        A2 = T.dot(B1, W2)
        B2 = A2 * (A2 > 0)
        A3 = T.dot(B2, W3)
        B3 = A3 * (A3 > 0)
        G = T.dot(B3, U)
        L1 = T.nnet.softmax(G)  # len(l) by num_label

        #L1=T.nnet.softmax(T.dot(T.tanh(T.dot(T.tanh(T.dot(T.tanh(T.dot(x,W1)),W2)),W3)),U))

        cost = T.nnet.categorical_crossentropy(L1, y).mean()
        gw1, gw2, gw3, gu = T.grad(cost, [W1, W2, W3, U])
        #gw_x = T.grad(cost, [x])

        log.info("Compiling theano model.")
        f1 = theano.function(inputs=[x, y, learn_rate], outputs=[cost], updates=(
            (W1, W1 - learn_rate * gw1), (W2, W2 - learn_rate * gw2), (W3, W3 - learn_rate * gw3),
            (U, U - learn_rate * gu)))

        #f2 = theano.function(inputs=[x, y], outputs=cost)
        prediction = T.argmax(L1, axis=1)
        discrepancy = prediction - y
        f3 = theano.function(inputs=[x, y], outputs=[discrepancy,prediction])
        #f4 = theano.function(inputs=[x, y], outputs=gw_x)

        alpha = self.alpha
        log.info("Read-in the training and test data.")
        open_train = open(train_file, "r")
        train_lines = open_train.readlines()
        open_test = open(test_file, "r")
        test_lines = open_test.readlines()

        log.info("Start training.")
        counter = 0
        start = time.time()
        iter_ = epochs
        for j in range(0, iter_):
            log.info("Epoch: {}...".format(j+1))
            x_ = []
            y_ = []
            for i in range(len(train_lines)):
                if i % 1000 == 0:
                    log.debug(i)
                counter = counter + 1
                current_alpha = alpha * (iter_ * len(train_lines) - counter) / (iter_ * len(train_lines))
                if current_alpha < 0.01: current_alpha = 0.01
                line_ = train_lines[i]
                G = line_.split("|")
                token_line = G[0]
                label_line = G[1]
                token_list = list(fromstring(token_line, dtype=int, sep=' '))
                x_ = self.contextwin(token_list)  # len(l) by win*dimension
                y_ = fromstring(label_line, dtype=int, sep=' ')
                f1(x_, y_, current_alpha)

            total_num = 0
            total_value = 0
            goldlabels = []
            predictions = []
            for i in range(len(test_lines)):
                line_ = test_lines[i]
                G = line_.split("|")
                token_line = G[0].strip()
                label_line = G[1].strip()

                y = fromstring(label_line, dtype=int, sep=' ')
                x = self.contextwin(list(fromstring(token_line, dtype=int, sep=' ')))
                total_num = total_num + x.shape[0]
                discrep, preds = f3(x, y)
                goldlabels.extend(list(y))
                predictions.extend(list(preds))
                total_value = total_value + x.shape[0] - count_nonzero(discrep)

            assert len(goldlabels) == len(predictions)
            log.info("f1 {}".format(f1_score(goldlabels, predictions, average="weighted")))
            acc = 1.00 * total_value / total_num
            log.info("acc " + str(acc))
        log.info("Training completed: {}s/epoch".format((time.time()-start)/iter_))


class ScoreExpEmbedding(ScoreEmbedding):
    def __init__(self, embedding_file, vocab_file, oov, hidden1, hidden2, hidden3, win, alpha, cembedding_file,
                 infer_side_win):
        super().__init__(embedding_file, vocab_file, oov, hidden1, hidden2, hidden3, win, alpha)
        self.embedding_matrix = self.load_embeddings(embedding_file)
        self.cembedding_matrix = load_npy(cembedding_file)
        self.infer_side_win = infer_side_win  # context size to each side for avgExp
        assert self.infer_side_win > 0
        assert self.cembedding_matrix.ndim == 2

        # use uniform avg for oov padding
        self.oov_vec = self.embedding_matrix[self.oov].mean(axis=0)
        self.oov_vecs = [self.oov_vec]*(self.win//2)

    def load_embeddings(self, embedding_file):
        embedding_matrix = load_npy(embedding_file)
        assert embedding_matrix.ndim == 3
        # avgExp (weighted mean)

        return embedding_matrix

    def contextwin(self, l):
        lpad = self.win // 2 * [self.oov] + l + self.win // 2 * [self.oov]
        linferpad = self.infer_side_win * [self.oov] + l + self.infer_side_win * [self.oov]
        # infer embeddings using sense expectations
        pivot_vecs = []
        for i in range(len(l)):
            # left and right contexts
            pivot_id = i+self.infer_side_win
            if linferpad[pivot_id] == self.oov:
                pivot_vecs.append(self.oov_vec)
                continue
            cs = linferpad[i:i+self.infer_side_win]+linferpad[(i + self.infer_side_win + 1):i+1+2*self.infer_side_win]
            cmean = mean(self.cembedding_matrix[array(cs)], axis=0)
            assert cmean.shape[0] == self.dimension
            act = dot(self.embedding_matrix[linferpad[pivot_id]], cmean)
            pivot_vecs.append(average(self.embedding_matrix[linferpad[pivot_id]], weights=softmax(act), axis=0))

        lpad_vecs = asarray(self.oov_vecs+pivot_vecs+self.oov_vecs)
        assert lpad_vecs.shape == (len(lpad), self.dimension)

        # prepare concatenated embeddings
        for i in range(len(l)):
            vector = lpad_vecs[i:(i + self.win)].reshape(1, self.win * self.dimension)
            if i == 0:
                x = vector
            else:
                x = concatenate((x, vector), axis=0)
        return x  # len(l) by win*dimension


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-alpha", type=float, default=0.025)
    parser.add_argument("-cembedding_file", help="In npy")
    parser.add_argument("-dev_file", default="../../../Datasets/wsj/wsjdev")
    parser.add_argument("-embedding_file", required=True, help="In npy")
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-hidden1", type=int, default=50)
    parser.add_argument("-hidden2", type=int, default=50)
    parser.add_argument("-hidden3", type=int, default=50)
    parser.add_argument("-oov", type=int, default=0)
    parser.add_argument("-tag_vocab_file", default="../../../Datasets/wsj/wsjtrain.tagvocab.json")
    parser.add_argument("-test_file", default="../../../Datasets/wsj/wsjtest")
    parser.add_argument("-train_file", default="../../../Datasets/wsj/wsjtrain")
    parser.add_argument("-vocab_file", required=True, help="Either in indexed json format or one word/line format")
    parser.add_argument("-vocab_limit", type=int, default=1000000)
    parser.add_argument("-vocab_limit_noexp", action="store_true",
                        help="Will use simple avg instead of weighted avg. for all words that exceed the args.vocab_limit (ie outside of vocab_limit-most frequent words.)")
    parser.add_argument("-win", type=int, default=7)
    parser.add_argument("-infer_side_win", type=int, default=3, help="Window size to each side during sense inference step")



    args = parser.parse_args()
    log.info("Settings: {}".format(args))
    log.info("Loading embeddings.")

    # context-sensitive inference for multisense embeddings
    if args.cembedding_file is not None:
        A = ScoreExpEmbedding(args.embedding_file, args.vocab_file, oov=args.oov, hidden1=args.hidden1,
                              hidden2=args.hidden2, hidden3=args.hidden3, win=args.win, alpha=args.alpha,
                              cembedding_file=args.cembedding_file, infer_side_win=args.infer_side_win)
    else:
        A = ScoreEmbedding(args.embedding_file, args.vocab_file, oov=args.oov, hidden1=args.hidden1,
                           hidden2=args.hidden2, hidden3=args.hidden3, win=args.win, alpha=args.alpha)  # use 0th, <s> symbol as oov

    log.info("Preparing training and test data.")
    train_file, len_tag_vocab = process(args.train_file, args.train_file+".bar", args.vocab_file, args.tag_vocab_file, args.vocab_limit)
    test_file, _ = process(args.test_file, args.test_file+".bar", args.vocab_file, args.tag_vocab_file, args.vocab_limit)

    A.Sequence_Level(train_file, test_file, len_tag_vocab, args.epochs)

