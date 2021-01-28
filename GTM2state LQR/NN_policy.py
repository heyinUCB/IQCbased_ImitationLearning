import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import inspect
import pickle
import matlab.engine
import logz
import scipy.linalg

# function for bulding up neural network
def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation,
                output_activation=None):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass)

        Hint: use tf.layers.dense
    """
    # raise NotImplementedError
    with tf.variable_scope(scope):
        sy_input = input_placeholder

        # Hidden layers
        hidden_layer = tf.layers.dense(inputs=sy_input,
                                        units=size,
                                        activation=activation,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), 
                                        # bias_initializer=tf.zeros_initializer()
                                        use_bias=False)
        for _ in range(n_layers - 1):
            hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                            units=size,
                                            activation=activation,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                            # bias_initializer=tf.zeros_initializer()
                                            use_bias=False)

        # Output layer
        output_placeholder = tf.layers.dense(inputs=hidden_layer,
                                            units = output_size,
                                            activation=output_activation,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), 
                                            # bias_initializer=tf.zeros_initializer()
                                            use_bias=False)
        return output_placeholder

def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def unnormalize(x, mean, std):
    return x * std + mean

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)

def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
        matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
        dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
        A matrix with the input matrices stacked along its main diagonal, having
        shape [..., \sum_i N_i, \sum_i M_i].

    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked

#############################################################################
#############################################################################
# define the agent
class Agent(object):
    def __init__(self, ob_dim, ac_dim, n_layers, batch_size, activation, sdp_var, iter, hyper_param, x1bound, x2bound):
        super(Agent, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.size = hyper_param["size"]
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.activation = activation
        self.sdp_var = sdp_var
        self.iter = iter
        self.rho = hyper_param["rho"]
        self.eta_NN = hyper_param["eta_NN"]
        self.x1bound = x1bound
        self.x2bound = x2bound

    def init_variable(self):
        # initialize the variables
        tf.get_default_session().run(tf.global_variables_initializer()) #pylint: disable=E1101
        self.saver = tf.train.Saver()
        # for v in tf.global_variables():
        #     if v.name == "nn_action/dense/kernel:0":
        #         self.W1 = tf.transpose(v)
        #         self.n1 = self.W1.shape[0]
        #     if v.name == "nn_action/dense_1/kernel:0":
        #         self.W2 = tf.transpose(v)
        #         self.n2 = self.W2.shape[0]
        #     if v.name == "nn_action/dense_2/kernel:0":
        #         self.W3 = tf.transpose(v)
        #         self.n3 = self.W3.shape[0]

    def fN_compute(self):
        N = block_diagonal([self.W1, self.W2, self.W3])
        Nvx = N[:self.n1+self.n2, :self.ob_dim]
        Nvw = N[:self.n1+self.n2, self.ob_dim:]
        Nux = N[self.n1+self.n2:, :self.ob_dim]
        Nuw = N[self.n1+self.n2:, self.ob_dim:]
        nphi = int(self.n1 + self.n2)
        Alpha = self.Alpha_compute()
        Beta = tf.eye(nphi)
        # intermediate = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))
        intermediate = tf.linalg.inv(tf.eye(nphi) - 1/2*tf.linalg.matmul(Nvw,Alpha+Beta))
        # fNvx = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvx;
        fNvx = tf.linalg.matmul(intermediate, Nvx)
        # fNvw = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);
        fNvw = 1/2*tf.linalg.matmul(tf.linalg.matmul(intermediate, Nvw), Beta-Alpha)
        # fNux = Nux + Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvx;
        fNux = Nux + tf.linalg.matmul(1/2*tf.linalg.matmul(Nuw,Alpha+Beta), fNvx)
        # fNuw = Nuw*1/2*(Beta-Alpha) + Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);
        fNuw = 1/2*tf.linalg.matmul(Nuw, Beta-Alpha) + tf.linalg.matmul(1/2*tf.linalg.matmul(Nuw, Alpha+Beta), fNvw)
        # fN = [fNux, fNuw;...
        #       fNvx, fNvw];
        fN = tf.concat([tf.concat([fNux, fNvx], 0), tf.concat([fNuw, fNvw], 0)], 1)
        return fN

    def Alpha_compute(self):
        w0up = tf.constant([[self.x1bound], [self.x2bound]])
        w0lb = -w0up
        v1up_list = []
        v1lb_list = []
        for i in range(self.n1):
            W1i = tf.reshape(self.W1[i,:], [1, self.ob_dim])
            v1up_list.append(1/2*tf.linalg.matmul(W1i, w0up+w0lb) + 1/2*tf.linalg.matmul(tf.math.abs(W1i), w0up-w0lb))
            v1lb_list.append(1/2*tf.linalg.matmul(W1i, w0up+w0lb) - 1/2*tf.linalg.matmul(tf.math.abs(W1i), w0up-w0lb))
        v1up = tf.reshape(tf.stack(v1up_list), [self.n1, 1])
        v1lb = tf.reshape(tf.stack(v1lb_list), [self.n1, 1])
        w1up = tf.math.tanh(v1up)
        w1lb = tf.math.tanh(v1lb)
        alpha1 = tf.math.minimum(tf.math.divide(w1up, v1up), tf.math.divide(w1lb, v1lb))
        alpha1 = tf.reshape(alpha1, [self.n1, ])
        v2up_list = []
        v2lb_list = []
        for i in range(self.n2):
            W2i = tf.reshape(self.W2[i,:], [1, self.n1])
            v2up_list.append(1/2*tf.linalg.matmul(W2i, w1up+w1lb) + 1/2*tf.linalg.matmul(tf.math.abs(W2i), w1up-w1lb))
            v2lb_list.append(1/2*tf.linalg.matmul(W2i, w1up+w1lb) - 1/2*tf.linalg.matmul(tf.math.abs(W2i), w1up-w1lb))
        v2up = tf.reshape(tf.stack(v2up_list), [self.n2, ])
        v2lb = tf.reshape(tf.stack(v2lb_list), [self.n2, ])
        w2up = tf.math.tanh(v2up)
        w2lb = tf.math.tanh(v2lb)
        alpha2 = tf.math.minimum(tf.math.divide(w2up, v2up), tf.math.divide(w2lb, v2lb))
        Alpha1 = tf.matrix_diag(alpha1)
        Alpha2 = tf.matrix_diag(alpha2)
        Alpha = block_diagonal([Alpha1, Alpha2])
        return Alpha

    def save_variables(self, logdir):
        self.saver.save(tf.get_default_session(), os.path.join(logdir, 'model.ckpt'))

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in actor critic
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        # raise NotImplementedError
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        learning_rate_ph = tf.placeholder(tf.float32, shape=[])

        return sy_ob_no, learning_rate_ph

    def build_computation_graph(self):
        """
            Notes on notation:

            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function

            Prefixes and suffixes:
            ob - observation
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)

            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.learning_rate_ph = self.define_placeholders()
        self.ac_prediction = build_mlp(
                                input_placeholder = self.sy_ob_no,
                                output_size = self.ac_dim,
                                scope = "nn_action",
                                n_layers=self.n_layers,
                                size=self.size,
                                activation=self.activation)
        # initialize the weights in the NN
        tf.get_default_session().run(tf.global_variables_initializer()) #pylint: disable=E1101
        for v in tf.global_variables():
            if v.name == "nn_action/dense/kernel:0":
                # self.W1 = tf.transpose(tf.convert_to_tensor(v))
                self.W1 = tf.transpose(v)
                self.n1 = self.W1.shape[0]
            if v.name == "nn_action/dense_1/kernel:0":
                # self.W2 = tf.transpose(tf.convert_to_tensor(v))
                self.W2 = tf.transpose(v)
                self.n2 = self.W2.shape[0]
            if v.name == "nn_action/dense_2/kernel:0":
                # self.W3 = tf.transpose(tf.convert_to_tensor(v))
                self.W3 = tf.transpose(v)
                self.n3 = self.W3.shape[0]
        self.ac_data = tf.placeholder(shape=[None, self.ac_dim], name="ac_true", dtype=tf.float32)
        if self.iter == 0: # nominal imitation learning
            self.loss = tf.losses.mean_squared_error(self.ac_data, self.ac_prediction)
        else: # safe imitation learning
            self.Q1 = np.array(self.sdp_var["Q1"],dtype='float32')
            self.Q2 = np.array(self.sdp_var["Q2"],dtype='float32')
            self.L1 = np.array(self.sdp_var["L1"],dtype='float32')
            self.L2 = np.array(self.sdp_var["L2"],dtype='float32')
            self.L3 = np.array(self.sdp_var["L3"],dtype='float32')
            self.L4 = np.array(self.sdp_var["L4"],dtype='float32')
            self.Yk = np.array(self.sdp_var["Yk"],dtype='float32')
            Q = scipy.linalg.block_diag(self.Q1, self.Q2)
            L = np.block([[self.L1, self.L2], [self.L3, self.L4]])
            self.fN = self.fN_compute()
            self.loss = self.eta_NN*tf.losses.mean_squared_error(self.ac_data, self.ac_prediction) + tf.linalg.trace(tf.matmul(np.transpose(self.Yk), tf.matmul(self.fN, Q)-L)) + self.rho/2*tf.math.square(tf.norm(tf.matmul(self.fN, Q)-L, 'fro', (0, 1)))
        self.update_op = tf.train.AdamOptimizer(self.learning_rate_ph).minimize(self.loss)

    def update_NN(self, ob_no, ac_na, learning_rate):
        tf.get_default_session().run([self.update_op], feed_dict={self.sy_ob_no: ob_no, self.ac_data: ac_na, self.learning_rate_ph: learning_rate})

    def compute_ac(self, ob_no):
        ac = tf.get_default_session().run([self.ac_prediction], feed_dict={self.sy_ob_no: ob_no})
        return ac

    def compute_loss(self, ob_no, ac_na):
        temp_loss = tf.get_default_session().run([self.loss], feed_dict={self.sy_ob_no: ob_no, self.ac_data: ac_na})
        return temp_loss

# solve the NN training problem using SGD
def solve_NNfit(ob_dim, ac_dim, n_layers, batch_size, activation, data, n_epoch, sdp_var, iter, hyper_param, logdir, x1bound, x2bound):
    # setup logger
    setup_logger(logdir, locals())

    agent = Agent(ob_dim, ac_dim, n_layers, batch_size, activation, sdp_var, iter, hyper_param, x1bound, x2bound)

    # tensorflow: config, session initialization
    tf_config = tf.ConfigProto(inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1)
    tf_config.gpu_options.allow_growth = True # may need if using GPU
    sess = tf.Session(config=tf_config)
    with sess:
        # build computaion graph
        agent.build_computation_graph()

        # tensorflow: variable initialization
        agent.init_variable()
        xu_data = data

        x_train = xu_data[:9500,:2]
        x_test = xu_data[9500:,:2]
        u_train = xu_data[:9500,2:]
        u_test = xu_data[9500:,2:]

        # number of points
        num_train_pt = x_train.shape[0]

        train_loss_list = []
        test_loss_list = []

        for epoch in range(n_epoch):
            rand_index = np.random.choice(num_train_pt, size=batch_size)
            x_batch = x_train[rand_index, :]
            u_batch = u_train[rand_index, :]

            learning_rate = 1e-3/(1 + 3*epoch/n_epoch)
            agent.update_NN(x_batch, u_batch, learning_rate)
            train_loss = agent.compute_loss(x_batch, u_batch)
            test_loss = agent.compute_loss(x_test, u_test)

            if epoch%4000 == 0:
                print("********** Iteration %i ************"%epoch)
                print('Trainting Loss = ', train_loss)
                print('Test Loss = ', test_loss)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            # if epoch%10000 == 0:
            #     agent.save_variables(logdir)
        agent.save_variables(logdir)
        logz.pickle_tf_vars()
        logz.save_params(hyper_param)
        if True:
            train_curve, = plt.plot(train_loss_list, 'r--', label='training Loss')
            test_curve, = plt.plot(test_loss_list, 'k--', label='test Loss')
            plt.legend(handles = [train_curve, test_curve])
            # plt.yscale('log')
            plt.xlabel('number of iterations')
            plt.ylabel('mean squared error')
            plt.title('NN policy training curve')
            plt.savefig(os.path.join(logdir, 'loss_vs_epoch'))
        W1 = agent.W1.eval()
        W2 = agent.W2.eval()
        W3 = agent.W3.eval()
    sess.close()
    tf.reset_default_graph()
    # return [agent.W1.eval(), agent.W2.eval(), agent.W3.eval()]
    return [W1, W2, W3]

def main():
    print(logz.colorize("Safe imitation learning begins", 'red', bold=True))
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'NN_policy' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    eng = matlab.engine.start_matlab()
    ob_dim = 2
    ac_dim = 1
    size = 16 # size of each hidden layer
    n_layers = 2 # number of hidden layers
    batch_size = 500
    n_epoch = 40000 # number of epochs for training the NN controller
    rho = 1
    eta_ROA = 5
    eta_NN = 100
    hyper_param = {"rho": rho, "eta_ROA": eta_ROA, "eta_NN": eta_NN, "size": size}
    activation = tf.nn.tanh
    x1bound = 2.0
    x2bound = 3.0

    # generate (x, u) data pairs
    xdata = []
    for i in range(100):
        for j in range(100):
            x = [(i-50)*0.2, (j-50)*0.2]
            xdata.append(x)
    xdata = np.array(xdata)
    K = np.array([[-0.276], [0.130]])
    udata = np.matmul(xdata, K)
    data = np.block([[xdata, udata]])

    # construct the dynamics for the GTM 2-state model
    # continuous-time dynamics xdot = Ac*x + Bc*u
    Ac = np.array([[-3.236, 0.923], [-45.34, -4.373]])
    Bc = np.array([[-0.317],[-59.989]])
    # sampling time
    dt = 0.02
    # discrete-time system
    AG = np.eye(ob_dim) + Ac*dt
    BG = Bc*dt
    nG = AG.shape[0]
    # initialize sdp_var
    sdp_var = {"Q1": np.zeros((nG,nG))}
    n_iter = 20 # number of iterations of safe imitation learning
    param = {}
    param["AG"] = matlab.double(AG.tolist())
    param["BG"] = matlab.double(BG.tolist())
    param["rho"] = matlab.double([rho])
    param["eta_ROA"] = matlab.double([eta_ROA])
    param["x1bound"] = matlab.double([x1bound])
    param["x2bound"] = matlab.double([x2bound])

    for i in range(n_iter):
        print(logz.colorize("safe learning iteration" + str(i), 'green', bold=True))
        # NN trainig step
        W1, W2, W3 = solve_NNfit(ob_dim, ac_dim, n_layers, batch_size, activation, data, n_epoch, sdp_var, i, hyper_param, os.path.join(logdir,'%d'%i),x1bound, x2bound)
        param["W1"] = matlab.double(W1.tolist())
        param["W2"] = matlab.double(W2.tolist())
        param["W3"] = matlab.double(W3.tolist())
        param["iter"] = matlab.int64([i])
        param["path"] = os.path.join(logdir,'%d'%i)
        # sdp step and Yk update step
        sdp_var = eng.solve_sdp(param)
        param["Yk"] = sdp_var["Yk"]

if __name__ == "__main__":
    main()