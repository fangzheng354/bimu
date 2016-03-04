import theano.tensor as TT

from bimu import losses

from bimu import optimizers
from bimu.initializations import *
from bimu.activations import sigmoid


#theano.config.floatX = "float64"
#theano.config.mode = "DebugMode"  # ‘ProfileMode’ (gpus), ‘DebugMode’, ‘FAST_RUN’, ‘FAST_COMPILE’
#theano.config.mode = "FAST_COMPILE"  # ‘ProfileMode’ (gpus), ‘DebugMode’, ‘FAST_RUN’, ‘FAST_COMPILE’
#theano.config.exception_verbosity = "high"
#theano.config.compute_test_value = "warn"

#theano.config.experimental.unpickle_gpu_on_cpu  # set to True for unpickling gpu build pickle on cpu


class FastSkipgram():
    def __init__(self, input_dim, emb_dim, weights=None):
        """
        :param input_dim: vocabulary size
        :param emb_dim: embedding dimension
        :param weights:
        """
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # different embeddings for pivot and context words
        self.W_w = uniform((input_dim, emb_dim), name="W_w")
        self.W_c = uniform((input_dim, emb_dim), name="W_c")
        #self.W_w = w2v((input_dim, emb_dim))
        #self.W_c = zeros((input_dim, emb_dim))

        self.params = [self.W_w, self.W_c]

        #self.activation = hard_sigmoid
        self.activation = sigmoid

    def build(self, loss="mse", optimizer="sgd", lr=0.01, rho=0.9, epsilon=1e-6):
        self.loss = losses.get(loss)
        optim = optimizers.get(optimizer, inst=False)
        if optim.__name__ == "RMSprop":
            self.optimizer = optim(lr=lr, rho=rho, epsilon=epsilon)
        elif optim.__name__ == "Adagrad":
            self.optimizer = optim(lr=lr, epsilon=epsilon)
        else:
            self.optimizer = optim(lr=lr)

        # get input to model
        self.X_train = TT.imatrix()  # n_pairs by 2

        self.y_pred, W_w_sel, W_c_sel = self.get_output()

        # for "gold"
        self.y = TT.zeros_like(self.y_pred)

        self.new_lr = TT.fscalar()

        train_loss = self.loss(self.y, self.y_pred)

        #updates = self.optimizer.get_updates_subtens(self.params, cost=train_loss)
        #updates = self.optimizer.get_updates_subtens(self.params, subparams=[W_w_sel, W_c_sel], cost=train_loss)
        updates = self.optimizer.get_updates(self.params, cost=train_loss)

        self.train = theano.function(inputs=[self.X_train, self.y],
                                     outputs=train_loss,
                                     updates=updates,
                                     allow_input_downcast=True,  # ignore mismatch between int and float dtypes
                                     mode=None)
        #theano.printing.pydotprint(self.train, outfile="symbolic_graph_unopt.png", var_with_name_simple=True)

    def get_output(self):
        W_w_sel = self.W_w[self.X_train[:, 0]]  # pivots
        W_c_sel = self.W_c[self.X_train[:, 1]]  # contexts

        dot = TT.sum(W_w_sel * W_c_sel, axis=1)
        dot = TT.reshape(dot, (self.X_train.shape[0], 1))

        return self.activation(dot), W_w_sel, W_c_sel  # for all pivot-context pairs


class Skipgram():
    def __init__(self, input_dim, emb_dim):
        """
        :param input_dim: vocabulary size
        :param emb_dim: embedding dimension
        """
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # different embeddings for pivot and context words
        self.W_w = uniform((input_dim+1, emb_dim), name="W_w")  # +1 because of padding
        self.W_c = uniform((input_dim+1, emb_dim), name="W_c")
        #self.W_w = w2v((input_dim+1, emb_dim))
        #self.W_c = zeros((input_dim+1, emb_dim))

        self.params = [self.W_w, self.W_c]

        #self.activation = hard_sigmoid
        self.activation = sigmoid

    def build(self, loss="mse", optimizer="sgd", lr=0.01, rho=0.9, epsilon=1e-6):
        self.loss = losses.get(loss)
        optim = optimizers.get(optimizer, inst=False)
        if optim.__name__ == "RMSprop":
            self.optimizer = optim(lr=lr, rho=rho, epsilon=epsilon)
        elif optim.__name__ == "Adagrad":
            self.optimizer = optim(lr=lr, epsilon=epsilon)
        else:
            self.optimizer = optim(lr=lr)

        # get input to model
        self.X_c = TT.imatrix(name="X_c")  # n_mbatches by n_contexts

        self.p = TT.ivector(name="p")  # n_pivots

        self.Y_pred, W_w_sel, W_c_sel = self.get_output()

        # for "gold"
        self.Y = TT.zeros_like(self.Y_pred)

        train_loss = self.loss(self.Y, self.Y_pred)
        #updates = self.optimizer.get_updates_subtens(self.params, cost=train_loss)
        #updates = self.optimizer.get_updates_subtens(self.params, subparams=[W_w_sel, W_c_sel], cost=train_loss)
        updates = self.optimizer.get_updates(self.params, cost=train_loss)

        self.train = theano.function(inputs=[self.p, self.X_c, self.Y],
                                     outputs=train_loss,
                                     updates=updates,
                                     allow_input_downcast=True,  # ignore mismatch between int and float dtypes
                                     mode=None)
        #theano.printing.pydotprint(self.train, outfile="symbolic_graph_unopt.png", var_with_name_simple=True)

    def get_output(self):
        W_p_sel = self.W_w[self.p][:, None, :]  # n_batches by dim
        W_c_sel = self.W_c[self.X_c]  # n_batches by contexts by dim
        #dot, _ = theano.scan(lambda p, W: TT.dot(W, p), sequences=[W_p_sel, W_c_sel])  # returns a list of arrays with dotproducts
        #same:
        #dot = TT.batched_dot(W_c_sel, W_p_sel)  # returns a list of arrays with dotproducts  # slow!
        dot = TT.sum(W_c_sel * W_p_sel, axis=2)

        return self.activation(dot), W_p_sel, W_c_sel  # for all pivot-context pairs