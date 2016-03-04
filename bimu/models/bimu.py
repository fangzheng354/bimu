import theano
import theano.tensor as TT

from bimu import losses, optimizers
from bimu.activations import sigmoid
from bimu.initializations import uniform


class Bimu():
    def __init__(self, input_dim, emb_dim, n_senses, W_w_f, lambdaF):
        """
        :param input_dim: vocabulary size
        :param emb_dim: embedding dimension
        :param W_w_f: generic, already trained l' embeddings
        """
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_senses = n_senses
        self.W_w_f = theano.shared(W_w_f, borrow=True)
        self.lambdaF = lambdaF  # contribution of l' context in sense prediction

        # multisense pivot params
        self.W_w = uniform((input_dim+1, n_senses, emb_dim), name="W_w")  # +1 because of padding
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
        # foreign language input
        self.X_c_f = TT.imatrix(name="X_c_f")  # n_batches*n_contexts

        self.X_c_f_mask = TT.matrix(dtype=theano.config.floatX, name="X_c_f_mask")  # n _batches*n_contexts

        #self.c_f_embs = TT.tensor3(dtype=theano.config.floatX, name="c_f_embs")
        self.context_f_means = self.context_f_mean(self.W_w_f[self.X_c_f])

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

        train_loss = self.get_loss(self.Y, self.Y_pred, mask=self.M_pad)

        #updates = self.optimizer.get_updates_subtens(self.params, cost=train_loss)
        #updates = self.optimizer.get_updates_subtens(self.params, subparams=[W_w_sel, W_c_sel], cost=train_loss)
        updates = self.optimizer.get_updates(self.params, cost=train_loss)

        self.train = theano.function(inputs=[self.p, self.X_c, self.X_c_f, self.X_c_f_mask, self.Y, self.max_y_is_1s,
                                             self.M_pad],
                                     #givens={self.c_f_embs: self.W_w_f[self.X_c_f]},
                                     #givens={self.context_f_means: self.context_f_mean(self.W_w_f[self.X_c_f])},
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
        sense_ids = self.predict_senses(W_p_sel, context_means, self.context_f_means)  # n_batches
        #pr = theano.printing.Print("")(sense_ids)
        # sense specific embs only
        W_sen = W_p_sel[TT.arange(W_p_sel.shape[0]), sense_ids]  # n_batches*emb_dim

        W_c_sel = self.W_c[self.X_c]  # n_batches by contexts by dim

        dot = TT.sum(W_c_sel * W_sen[:, None, :], axis=2)  # n_batches*n_contexts

        return self.activation(dot)  # for all pivot-context pairs

    def context_mean(self):
        """
        C_i
        :param Y: use to mask negative and padded contexts
        :param max_y: use in mean (to normalize the sum)
        """
        c_embs = self.W_c[self.X_c]  # (n_batches.n_contexts)*emb_dim

        C_embs = c_embs * self.Y[:, :, None]  # all irrelevant to zero; n_batches*n_contexts*emb_dim

        return TT.sum(C_embs, axis=1) / self.max_y_is_1s.reshape((-1, 1))  # n_batches*emb_dim

    def context_f_mean(self, c_f_embs):
        """
        C'_i
        """
        C_f_embs = c_f_embs * self.X_c_f_mask[:, :, None]  # all irrelevant to zero; n_batches*n_contexts*emb_dim
        max_y_is_1s = TT.sum(self.X_c_f_mask, axis=1)
        max_y_is_1s.name = "max_y_is_1s"
        # 0-->1 to get 0-emb mean vector when no context is available:
        max_y_is_1s = TT.switch(TT.eq(max_y_is_1s, 0), 1, max_y_is_1s)
        max_y_is_1s.name = "max_y_is_1s_switched"
        r = max_y_is_1s.reshape((-1, 1))
        r.name = "reshaped"

        return TT.sum(C_f_embs, axis=1) / r  # n_batches*emb_dim

    def predict_senses(self, W, C, C_f):
        """
        :param W: n_batches * n_senses * emb_dim
        :param C: context means; n_batches * emb_dim
        :param C_f: context means; n_batches * emb_dim, can be 0-vector
        """
        p = TT.sum(W * ((1.-self.lambdaF)*C + self.lambdaF*C_f)[:, None, :], axis=2)  # n_pairs*n_senses, row-stochastic

        return p.argmax(axis=1)  # n_batches

    def get_loss(self, Y, Y_pred, mask):
        return self.loss(Y, Y_pred, mask=mask)