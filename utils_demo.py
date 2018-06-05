import os
import numpy as np
import tensorflow as tf
import cv2

# +-* + () + 10 digit + blank + space
num_classes = 3 + 2 + 10 + 1 + 1

maxPrintLen = 100

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 60, 'image height')
tf.app.flags.DEFINE_integer('image_width', 180, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('max_stepsize', 64, 'max stepsize in lstm, as well as '
                                                'the output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 40, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', './imgs/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './imgs/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './imgs/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'num of gpus')

FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

charset = '0123456789+-*()'
encode_maps = {}
decode_maps = {}

# 构造字典
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

class DataIterator:
    """docstring for DataIterator"""
    def __init__(self, data_dir):
        super(DataIterator, self).__init__()
        self.image = [] # [[im1], [im2], [im3]],[im1] = [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel]
        self.labels = [] # [[123], [231], [492]...]
        # 遍历train_folder
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                image_name = os.path.join(root, file_path)
                # 导入灰度图
                # 255.是float格式
                im = cv2.imread(image_name, 0).astype(np.float32) / 255.
                # 将图片统一到一样的高度
                im = tf.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                # 构造labels:[index1, index2...]
                # 图片被命名为：/.../<folder>/00000_abcd.png
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                # code:'abcd'->'1234'
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                self.labels.append(code)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    # 生成batch
    def input_index_generate_batch(self, index = None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:    
            image_batch = self.image
            label_batch = self.labels

            # 定义一个内部方法
            # lengths:RNNRUN参数用来限制每个batch的输出长度
            # 这里选择每个样本的LSTM输出都是
        def get_input_lens(sequences):
            # CNN最后一层outchannels是64，也是timestep
            lengths = np.asarray([FLAGS.max_stepsize for _ in sequences], dtype = np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels

# 创建一个真值标签的序列和其indices
def sparse_tuple_from_label(sequences, dtype = np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = [] # [(0,0), (0,1), (1,0), (1,1), (1,2)]
    values = [] # [1, 3, 4, 4, 2]

    # zip:将list之间以元祖的方式结合起来，返回tuple的list
    # extend:Append items from iterable to the end of the array.
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype = np.int64)
    values = np.asarray(values, dtype = dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype = dtype)
    # 最后的shape:[num_words_in_labels, max_len_word]

    return indices, values, shape



