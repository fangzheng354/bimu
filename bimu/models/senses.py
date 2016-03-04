import theano
import theano.tensor as TT

from bimu import losses, optimizers
from bimu.activations import sigmoid
from bimu.initializations import uniform, zeros


class Senses():
    def __init__(self, input_dim, emb_dim, n_senses, W_w=None, W_c=None):
        """
        :param input_dim: vocabulary size
        :param emb_dim: embedding dimension
        :param n_senses:
        :param W_w: pretrained params
        :param W_c: pretrained params
        """
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_senses = n_senses

        # multisense pivot params
        if W_w is not None:
            self.W_w = theano.shared(W_w)
        else:
            self.W_w = uniform((input_dim+1, n_senses, emb_dim), name="W_w")  # +1 because of padding
        if W_c is not None:
            self.W_c = theano.shared(W_c)
        else:
            self.W_c = uniform((input_dim+1, emb_dim), name="W_c")

        self.params = [self.W_w, self.W_c]
        self.activation = sigmoid

    def build(self, loss="crossentropy", optimizer="rmsprop", lr=0.01, rho=0.9, epsilon=1e-6):
        self.loss = losses.get(loss+"_masked")
        optim = optimizers.get(optimizer, inst=False)

        if optim.__name__ == "RMSprop":
            self.optimizer = optim(lr=lr, rho=rho, epsilon=epsilon)
        elif optim.__name__ == "Adagrad":
            self.optimizer = optim(lr=lr, epsilon=epsilon)
        else:
            self.optimizer = optim(lr=lr)

        # get input to model
        self.X_c = TT.imatrix(name="X_c")  # n_batches*n_contexts
        self.p = TT.ivector(name="p")  # pivots (n_batches)
        self.max_y_is_1s = TT.vector(dtype=theano.config.floatX, name="max_y")  # n_batches
        self.M_pad = TT.matrix(dtype=theano.config.floatX, name="mask_pad")  # mask for padded; n_batches*n_contexts

        # declare symbolic vars for encoder
        # context ids for pivots in a pair from X_train: n_pairs*n_contexts(variable)
        # float matrix containing nan for out of sent context ids
        # convert to int during indexing the embeddings
        #self.context_idx = TT.imatrix(name="context_idx")

        self.Y = TT.matrix(dtype=theano.config.floatX, name="Y")  # n_batches*n_contexts
        # get input to model
        self.Y_pred = self.get_output()  # Y_pred: n_batches*n_contexts

        #pr = theano.printing.Print("")(self.Y_pred)
        train_loss = self.get_loss(self.Y, self.Y_pred, mask=self.M_pad)

        #updates = self.optimizer.get_updates_subtens(self.params, cost=train_loss)
        #updates = self.optimizer.get_updates_subtens(self.params, subparams=[W_w_sel, W_c_sel], cost=train_loss)
        updates = self.optimizer.get_updates(self.params, cost=train_loss)

        self.train = theano.function(inputs=[self.p, self.X_c, self.Y, self.max_y_is_1s, self.M_pad],
                                     on_unused_input='warn',
                                     outputs=train_loss,
                                     updates=updates,
                                     allow_input_downcast=True,  # ignore mismatch between int and float dtypes
                                     mode=None
                                     # mode=theano.compile.MonitorMode(post_func=detect_nan)
                                     #mode=theano.compile.MonitorMode(post_func=detect_nan).excluding('local_elemwise_fusion', 'inplace')
                                     )
        #theano.printing.pydotprint(self.train, outfile="symbolic_graph_unopt.png", var_with_name_simple=True)

    def get_output(self):
        # argmax of dots
        # obtain context mean embeddings
        context_means = self.context_mean()
        # Neelakantan different: argmax with cosine and cluster mean
        W_p_sel = self.W_w[self.p]  # n_batches*n_senses*emb_dim
        sense_ids = self.predict_senses(W_p_sel, context_means)  # n_batches
        # sense specific embs only
        W_sen = W_p_sel[TT.arange(W_p_sel.shape[0]), sense_ids]  # n_batches*emb_dim

        W_c_sel = self.W_c[self.X_c]  # n_batches by contexts by dim

        dot = TT.sum(W_c_sel * W_sen[:, None, :], axis=2)  # n_batches*n_contexts
        return self.activation(dot)  # for all pivot-context pairs

    def context_mean(self):
        """
        :param Y: use to mask negative and padded contexts
        :param max_y: use in mean (to normalize the sum)
        """
        c_embs = self.W_c[self.X_c]  # (n_batches.n_contexts)*emb_dim
        C_embs = c_embs * self.Y[:, :, None]  # all irrelevant to zero; n_batches*n_contexts*emb_dim

        return TT.sum(C_embs, axis=1) / self.max_y_is_1s.reshape((-1, 1))  # n_batches*emb_dim

    def predict_senses(self, W, C):
        """
        :param W: n_batches * n_senses * emb_dim
        :param C: context means; n_batches * emb_dim
        """
        #p = TT.nnet.softmax(TT.sum(W * C[:, None, :], axis=2) + self.Wb[self.p])  # n_pairs*n_senses, row-stochastic
        #p = TT.nnet.softmax(TT.sum(W * C[:, None, :], axis=2))  # n_pairs*n_senses, row-stochastic
        #np.argmax(np.sum(W_shared*C_shared[:,None,:], axis=2), axis=1)
        p = TT.sum(W * C[:, None, :], axis=2)  # n_pairs*n_senses, row-stochastic
        return p.argmax(axis=1)  # n_batches

    def get_loss(self, Y, Y_pred, mask):
        return self.loss(Y, Y_pred, mask=mask)


class SensesInference(Senses):
    def __init__(self, input_dim, emb_dim, n_senses, W_w=None, W_c=None):
        super().__init__(input_dim, emb_dim, n_senses, W_w=W_w, W_c=W_c)
        self.activation = None

    def build_inference(self):
        # get input to model
        self.X_c = TT.imatrix(name="X_c")  # n_batches*n_contexts
        self.p = TT.ivector(name="p")  # pivots (n_batches)
        self.max_y_is_1s = TT.vector(dtype=theano.config.floatX, name="max_y")  # n_batches
        self.M_pad = TT.matrix(dtype=theano.config.floatX, name="mask_pad")  # mask for padded; n_batches*n_contexts
        self.Y = TT.matrix(dtype=theano.config.floatX, name="Y")  # n_batches*n_contexts

        # get input to model
        s = self.get_output()  # n_batches for SensesInference; n_batches*n_senses for SensesExpectationInference
        self.train = theano.function(inputs=[self.p, self.X_c, self.Y, self.max_y_is_1s],
                                     on_unused_input='warn',
                                     outputs=[self.p, s],
                                     #updates=[update_weightsum, update_countsum],
                                     allow_input_downcast=True,  # ignore mismatch between int and float dtypes
                                     mode=None)

    def get_output(self):
        # argmax of dots
        # obtain context mean embeddings
        context_means = self.context_mean()
        W_p_sel = self.W_w[self.p]  # n_batches*n_senses*emb_dim
        sense_ids = self.predict_senses(W_p_sel, context_means)  # n_batches

        return sense_ids


class SensesExpectationInference(SensesInference):
    def __init__(self, input_dim, emb_dim, n_senses, W_w=None, W_c=None):
        super().__init__(input_dim, emb_dim, n_senses, W_w=W_w, W_c=W_c)
        self.activation = None

    def get_output(self):
        # argmax of dots
        # obtain context mean embeddings
        context_means = self.context_mean()
        W_p_sel = self.W_w[self.p]  # n_batches*n_senses*emb_dim
        sense_exps = self.predict_senses(W_p_sel, context_means)  # n_batches*n_senses

        return sense_exps#, context_means, W_p_sel

    def predict_senses(self, W, C):
        """
        :param W: n_batches * n_senses * emb_dim
        :param C: context means; n_batches * emb_dim
        """
        expects = TT.nnet.softmax(TT.sum(W * C[:, None, :], axis=2))  # n_batches*n_senses, row-stochastic

        return expects

