import tensorflow as tf
import utils
from tensorflow.python.training import moving_averages

FLAGS = utils.FLAGS
num_classes = utils.num_classes

class LSTMOCR(object):
    """docstring for LSTMOCR"""
    def __init__(self, arg):
        super(LSTMOCR, self).__init__()
        self.mode = mode
        # image: 4d
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # 稀疏tensor：用于ctc_loss的op
        # OCR的结果是不定长的，所以label实际上是一个稀疏矩阵
        self.labels = tf.sparse_placeholder(tf.int32)
        # size [batch_size] 1d
        self.seq_len = tf.placeholder(tf.int32, [None])
        # 
        self._extra_train_ops = []

    # 创建图
    def build_graph(self):
        self._build_model()
        self._build_train_op()
        # 可视化
        self.merged_summary = tf.summary.merge_all()

        # 构建模型的图
    def _build_model(self):
        # max_stepsize是timestep
        filters = [64, 128, 128, FLAGS.max_stepsize]
        strides = [1, 2]

        # CNN part
        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                # in_channels：1 ocr的输入图像为灰度图
                x = self._conv2d(self.inputs, 'con-1', 3, 1, filters[0], strides[0])
                x = self._batch_norm('bn1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2', 3, filters[0], filters[1], strides[0])
                x = self._batch_norm('bn2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.VariableScope('unit-3'):
                x = self._conv2d(x, 'con-3', 3, filters[1], filters[2], strides[0])
                x = self._batch_norm('bn3', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-4'):
                x = self._conv2d(x, 'cnn-4', 3, filters[2], filters[3], strides[0])
                x = self._batch_norm('bn-4', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

        # LSTM part
        with tf.variable_scope('lstm'):
            # 将feature_maps向量化，变成3d：[batchsize, w*h, outchannels]
            # filters[3]是timestep，也是最后一层cnn的outchannels
            # 每个lstm单元的输入是一个特征纤维，这一点是和ctpn一样的，只不过ctpn的timestep是w行
            # tf.nn.sigmoid_cross_entropy_with_logits()
            x = tf.reshape(x, [FLAGS.batch_size, -1, filters[3]])
            x = tf.transpose(x, [0, 2, 1])
            # timestep:64
            # batch_size * 64 * 48
            x.set_shape([FLAGS.batch_size, filters[3], 48])

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple = True)
            # 训练模式下多层LSTM之间添加drop_out层
            if self.mode == 'train':
                cell = tf.contrib.rnn.DropoutWrapper(cell = cell, out_keep_prob = 0.8)
            
            # 添加一层LSTM 
            cell1 = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple = True)
            if self.mode == 'train':
                cell1 = tf.contrib.rnn.DropoutWrapper(cell = cell1, out_keep_prob = 0.8)

            # 堆叠多层LSTM
            stack = contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple = True)
            # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
            # dynamic_rcnn的sequence_length参数用来限制每个batch的输出长度
            # outputs是RNNcell的输出
            outputs, _ = tf.nn.dynamic_rnn(stack, x, self.seq_len, dtype = tf.float32)

            # reshape outputs
            # 执行线性输出
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            W = tf.get_variable(name = 'W', shape = [FLAGS.num_hidden, num_classes], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', shape = [num_classes], dtype = float32, initializer = tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b

            # 重新reshape到输入的尺寸3d
            shape = tf.shape(x)
            self.logits = tf.reshape(x, [shape[0], -1, num_classes])
            # ctc_loss()time_major参数：time_major = True
            self.logits = tf.transpose(x, [1, 0, 2])

            # 构建训练op
    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable = False)

        # 计算CTC损失值:softmax操作
        # input:[batch_size,max_time_step,num_classes]
        self.loss = tf.nn.ctc_loss(labels = self.labels, inputs = self.logits, sequence_length = self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, self.global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase = True)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)

        # 参数更新方法采用Adam
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.initial_learning_rate, beta1 = FLAGS.beta1, beta2 = FLAGS.beta2).minimize(self.loss, global_step = self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        # group:将语句变成操作
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        # 
        # 对输入的logits执行beam search 解码
        #  如果 merge_repeated = True, 在输出的beam中合并重复类
        #  返回的self.decoded是稀疏矩阵
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated = False)
        # 将序列的稀疏矩阵转成dense矩阵
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value = -1)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        # variable_scope实现共享变量
        with tf.variable_scope(name):
            kernel = tf.get_variable(name = 'DW', shape = [filter_size, filter_size, in_channels, out_channels], dtype = tf.float32, initializer = contrib.layers.xavier_initializer())

            b = tf.get_variable(name = 'bias', shape = [out_channels], dtype = tf.float32, initializer = tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding = 'SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32, initializer = tf.constant_initializer())
            gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer = tf.constant_initializer())

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name = 'moments')

                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32, initializer = tf.constant_initializer())
                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32, initializer = tf.constant_initializer())

                # 滑动平均法计算mean和variance
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable('moving_mean', params_shape, tf.float32, initializer = tf.constant_initializer(), trainable = False)
                variance = tf.get_variable('moving_variace', params_shape, tf.float32, initializer = tf.constant_initializer(), trainable = False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness = 0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name = 'leaky_relu')

    def _max_pool(self, x, ksize, strides):
            return tf.nn.max_pool(x, ksize = [1, ksize, ksize, 1], strides = [1, strides, strides, 1], padding = 'SAME', name = 'max_pool')
