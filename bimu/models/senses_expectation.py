import theano
import theano.tensor as TT

from bimu.initializations import zeros
from bimu.models.senses import Senses


class SensesExpectation(Senses):
    def __init__(self, input_dim, emb_dim, n_senses, lambdaH, lambdaL2, adjust):
        super().__init__(input_dim, emb_dim, n_senses)
        self.Wb = zeros((input_dim+1, n_senses), name="Wb")  # sense- and word-specific bias
        self.H = TT.fscalar()  # entropy
        self.L2 = TT.fscalar()
        self.lambdaH = lambdaH  # weight for entropy regularizer
        self.lambdaL2 = lambdaL2  # weight for L2 regularizer
        if lambdaL2 == 0.:
            self.L2 = 0.
        else:
            self.L2 = TT.sum(TT.sqr(self.W_w)) + TT.sum(TT.sqr(self.W_c))
        self.adjust = adjust
        #self.params += [self.Wb]

    def get_output(self):
        """
        Scan over senses (very low dim.)
        """
        def dot_sense(i, W_c_sel, W_sen):
            s = W_sen[:, i, :]
            return TT.sum(W_c_sel * s[:, None, :], axis=2)

        # obtain context mean embeddings
        context_means = self.context_mean()
        W_p_sel = self.W_w[self.p]  # n_batches*n_senses*emb_dim
        sense_expect, self.H = self.predict_senses(W_p_sel, context_means)  # n_batches * n_senses
        # weight contribution of each sense
        W_sen = W_p_sel * sense_expect[:, :, None]  # n_batches*n_senses*emb_dim

        W_c_sel = self.W_c[self.X_c]  # n_batches*n_contexts*dim (3*4*3)

        #dot_, _ = theano.scan(lambda C, S: TT.dot(C, S.transpose()), sequences=[W_c_sel, W_sen], outputs_info=None)
        #dot = TT.sum(dot_, axis=2)

        dot_, _ = theano.scan(dot_sense, sequences=TT.arange(self.n_senses), non_sequences=[W_c_sel, W_sen], outputs_info=None)
        dot = TT.sum(dot_, axis=0)

        return self.activation(dot)  # for all pivot-context pairs

    def predict_senses(self, W, C):
        """
        :param W: n_batches * n_senses * emb_dim
        :param C: context means; n_batches * emb_dim
        """
        #expects = TT.nnet.softmax(TT.sum(W * C[:, None, :], axis=2) + self.Wb[self.p])  # n_pairs*n_senses, row-stochastic
        expects = TT.nnet.softmax(TT.sum(W * C[:, None, :], axis=2))  # n_pairs*n_senses, row-stochastic
        # -np.sum(expects*np.log(expects), axis=1)
        H = -TT.sum(expects * TT.log(expects))
        return expects, H

    def get_loss(self, Y, Y_pred, mask):
        return self.loss(Y, Y_pred, mask=mask) + self.L2*self.lambdaL2*self.adjust - self.H*self.lambdaH  # add entropy as regularization