import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
import utils
import helper

# 使用gpu0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = utils.FLAGS

logger = logging.getLogger('Train for OCR using CNN+LStM+CTC')
logger.setLevle(logging.INFO)

# 开始训练模型
def train(train_dir = None, val_dir = None, mode = 'train'):
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    # 创建图
    model.build_graph()

    print('loading train data, please wait---------------------')
    # 训练数据构造器
    train_feeder = utils.DataIterator(data_dir = train_dir)
    print('get image:', train_feeder.size)
    print('loading validation data, please wait---------------------')
    # 验证数据构造器
    val_feeder = utils.DataIterator(data_dir = val_dir)
    print('get image:', val_feeder.size)

    num_train_samples = train_feeder.size # 100000
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size) # example: 100000 / 100

    num_val_samples = val_feeder.size # 100000
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size) # example: 100000 / 100
    # 随机打乱验证集样本
    shuffle_idx_val = np.random.permutation(num_val_samples)

    with tf.device('/gpu:0'):
        # tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
        # allow_soft_placement = True
        # 如果你指定的设备不存在，允许TF自动分配设备
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())

            # 创建saver对象，用来保存和恢复模型的参数
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)
            # 将sess里的graph放到日志文件中
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            # 如果之前有保存的模型参数，将之恢复到现在的sess中
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))

             print('=============================begin training=============================')
             # 开始训练
             for cur_epoch in range(FLAGS.num_epochs):
                 shuffle_idx = np.random.permutation(num_train_samples)
                 train_cost = 0
                 start_time = time.time()
                 batch_time = time.time()

                 for cur_batch in range(num_batches_per_epoch):
                     if (cur_batch + 1) % 100 == 0:
                        print('batch', cur_batch, ':time', time.time() - batch_time)
                    batch_time = time.time()
                    # 构造当前batch的样本indexs
                    # 在训练样本空间中随机选取batch_size数量的的样本
                    indexs = [shuffle_idx[i % num_train_samples] for i in range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                    batch_inputs, batch_seq_len, batch_labels = train_feeder.input_index_generate_batch(indexs)
                    # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
                    # 构造模型feed参数
                    feed = {model.inputs:batch_inputs, model.labels: batch_labels, model.seq_len: batch_seq_len}

                    # 执行图
                    # fetch操作取回tensors
                    summar_str, batch_cost, step, _ = sess.run([model.merged_summay, model.cost, model.global_step, model.train_op], feed)
                    # 计算损失值
                    # 这里的batch_cost是一个batch里的均值
                    train_cost += batch_cost * FLAGS.batch_size
                    # 可视化
                    train_writer.add_summary(summar_str, step)

                    # 保存模型文件checkpoint
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save the checkpoint of{0}', format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step = step)

                    # 每个batch验证集上得到解码结果
                    if step % FLAGS.validation_steps == 0:
                        acc_batch_total = 0
                        lastbatch_err = 0
                        lr = 0
                        # 得到验证集的输入
                        # 每个batch做迭代验证
                        for j in xrange(num_batches_per_epoch_val):
                            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                            val_inputs, val_seq_len, val_labels = val_feeder.input_index_generate_batch(indexs_val)
                            val_feed = {model.inputs: val_inputs, modell.labels: val_labels, model.seq_len: val_seq_len}

                            dense_decoded, lastbatch_err, lr = sess.run([model.dense_decoded, model.lrn_rate], val_feed)

                            # 打印在验证集上返回的结果
                            ori_labels = val_feeder.the_label(indexs_val)
                            acc = utils.accuracy_calculation(ori_labels, dense_decoded, ignore_value = -1, isPrint = True)
                            acc_batch_total += acc

                        accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples

                        avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)
                        # train_err /= num_train_smaples
                        
                        now = datetime.datetime.time()
                        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                              "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                              "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                        print(log.format(now.month, now.day, now.hour, now.minute, now.second, cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost, lastbatch_err, time.time() - start_time, lr))

# 测试集
def infer(img_path, mode = 'infer'):
    # imgList = load_img_path('/home/yang/Downloads/FILE/ml/imgs/image_contest_level_1_validate/')
    imgList = helper.load_img_path(img_path)
    print(imgList[:5])

    model = cnn_lstm_otc_ocr(mode)
    model = build_graph()

    total_steps = len(imgList) / FLAGS.batch_size

    cofig = ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables_initializer(), max_to_keep = 100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        decoded_expression = []
        for curr_step in xrange(total_steps):
            imgs_input = []
            seq_len_input = []
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                im = cv2.imread(img, 0).astype(np.float32) / 255.
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

                # 返回seq_len参数
                def get_input_lens(seqs):
                    length = np.array([FLAGS.max_stepsize for _ in seqs], dtype = np.int64)

                    return seqs, length

                inp, seq_len = get_input_lens(np.array([im]))
                imgs_input.append(im)
                seq_len_input.append(seq_len)

        imgs_input = np.asarray(imgs_input)
        seq_len_input = np.asarray(seq_len_input)
        seq_len_input = np.reshape(seq_len_input, [-1])

        feed = {model.inputs: imgs_input, model.seq_len: seq_len_input}
        dense_decoded_code = sess.run(model.dense_decoded, feed)

        for item in dense_decoded_code:
            expression = ''

            for i in item:
                if i == -1:
                    expression += ''
                else:
                    # 从label字典中找到相应的字母label
                    expression += utils.decode_maps[i]
                decoded_expression.append(expression)

    with open('./result.txt', 'a') as f:
        for code in decoded_expression:
            f.write(code + '\n')

def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    else FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)

        elif FLAGS.mode == 'infer':
            infer(FLAGS.infer_dir, FLAGS.mode)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()