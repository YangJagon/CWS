# encoding=utf-8
# Project: transfer_cws
# Author: xingjunjie
# Create Time: 30/11/2017 9:15 AM on PyCharm

import tensorflow as tf
from utils import Progbar
from data_utils import pad_sequences, minibatches, get_chunks, minibatches_evaluate
import numpy as np
import os
from functools import partial
from penalty import MKL, CMD, MMD, gaussian_kernel_matrix, _de_pad


class Model(object):
    def __init__(self, args, ntags, nwords, ntarwords=None, src_embedding=None,
                 target_embedding=None, logger=None, src_batch_size=None):
        self.args = args
        self.src_embedding = src_embedding  # None
        self.target_embedding = target_embedding  # None
        self.ntags = ntags
        self.nwords = nwords
        self.ntarwords = ntarwords
        self.logger = logger
        self.init_lr = args.learning_rate  # 默认0.01
        self.src_batch_size = src_batch_size    # 19
        if(not self.src_batch_size is None):
            self.target_batch_size = self.args.batch_size - self.src_batch_size # 1
        else:
            self.target_batch_size = 1
            self.src_batch_size = 0

        self.describe = "parallel training, only with mmd, model-1"

        # xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
        # 该函数返回一个用于初始化权重的初始化程序 “Xavier” 。
        # 这个初始化器是用来保持每一层的梯度大小都差不多相同。
        # uniform: 使用uniform或者normal分布来随机初始化。
        # seed: 可以认为是用来生成随机数的seed
        # dtype: 只支持浮点数。
        self.initializer = tf.contrib.layers.xavier_initializer()

        # 用于正则化的函数，参数是权重，默认为 0.1
        # l2是平方和（如(1²+(-2)²+(-3)²+4²)*0.5），l1是绝对值和
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(self.args.l2_ratio)

        self.info = {
            'dev': [],
            'train': [],
            'loss': [],
            'test': None
        }

    def add_placeholder(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            # tf.placeholder(dtype, shape=None, name=None) --> 占位符
            # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
            # shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
            # name：名称

            # batch size
            self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

            # 每一行应为一独热向量，指示整个词向量空间中的哪一个
            # shape = [batch size, max length of sequence in batch]
            self.src_word_ids = tf.placeholder(tf.int32, shape=[None, None], name='src_word_ids')
            # shape = [batch size, max length of sequence in batch]
            self.target_word_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_word_ids')

            # 每一行应为每条句子的实际长度，用于lstm的timestep
            # shape = [batch size]
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
            # shape = [batch size]
            self.src_sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='src_sequence_lengths')
            # shape = [batch size]
            self.target_sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='target_sequence_lengths')

            # 每次输入的真实标签集合
            # shape = [batch size, max length of sequence in batch]
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

            # hyper parameters
            # 丢弃率，防止过拟合
            self.dropout = tf.placeholder(tf.float32, shape=[], name='dropout')
            # 学习率，调整梯度
            self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
            # 应为指示是否在训练
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def get_feed_dict(self, sentences, labels, target_words, lr=None, dropout=None, src_batch_size=None, mode="all",
                      is_training=True):
        # sentences: src_words
        if mode == 'all':
            # 返回两个数组，all_words_ids填充0为 batchSize * maxLength的数组，sequence_lengths为各句子的长度
            all_words_ids, sequence_lengths = pad_sequences(sentences + target_words, pad_tok=0)
            # 将源域和目标域的数据分开
            words_ids = all_words_ids[:src_batch_size] + [[0] * len(all_words_ids[0])] * (
                    self.args.batch_size - src_batch_size)
            src_sequence_lengths = sequence_lengths[:src_batch_size] + [0] * (self.args.batch_size - src_batch_size)
            target_words_ids = [[0] * len(all_words_ids[0])] * src_batch_size + all_words_ids[src_batch_size:]
            target_sequence_lengths = [0] * src_batch_size + sequence_lengths[src_batch_size:]

            feed_dict = {
                self.src_word_ids: words_ids,
                self.src_sequence_lengths: src_sequence_lengths,
                self.target_word_ids: target_words_ids,
                self.target_sequence_lengths: target_sequence_lengths,
                self.sequence_lengths: sequence_lengths,
                self.batch_size: self.args.batch_size,
                self.is_training: is_training,
            }
        elif mode == 'target':
            # target 模式源域数据全置零
            # target_sequence_lengths = len(target_words)
            # target_words_ids = [[0] * len(target_words)] * src_batch_size + target_words
            # target_sequence_lengths = [0] * src_batch_size + [len(target_words)]
            target_words_ids, target_sequence_lengths = pad_sequences(target_words, pad_tok=0)
            sequence_lengths = target_sequence_lengths
            feed_dict = {
                self.src_word_ids: np.zeros_like(target_words_ids),
                self.src_sequence_lengths: np.zeros_like(target_sequence_lengths),
                self.target_word_ids: target_words_ids,
                self.target_sequence_lengths: target_sequence_lengths,
                self.sequence_lengths: target_sequence_lengths,
                self.batch_size: self.args.batch_size,
                self.is_training: is_training,
            }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed_dict[self.labels] = labels

        if lr is not None:
            feed_dict[self.lr] = lr

        if dropout is not None:
            feed_dict[self.dropout] = dropout

        return feed_dict, sequence_lengths

    # 添加源域词向量空间的选取器
    def add_src_word_embeddings_op(self):
        # 指明运行在哪块设备上
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            # 上下文管理器，用于定义创建变量(或层)的操作。
            with tf.variable_scope("src_word"):
                _word_embeddings = tf.get_variable('embedding', shape=[self.nwords, self.args.embedding_size],
                                                   initializer=self.initializer,
                                                   trainable=not self.args.disable_src_embed_training,
                                                   regularizer=self.l2_regularizer)
                # tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素
                # tf.nn.embedding_lookup(tensor,id)：即tensor就是输入的张量，id 就是张量对应的索引
                word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.src_word_ids)

                # source domain 和 target domain 是否分享词向量
                if self.args.share_embed:
                    target_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.target_word_ids)
                    self.target_word_embeddings = tf.nn.dropout(target_word_embeddings, self.dropout)

            if self.args.use_pretrain_src:
                pre_train_size = self.src_embedding.shape[0]
                self.src_embedding_init = _word_embeddings[:pre_train_size].assign(self.src_embedding)

            self.src_word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    # 添加目标域词向量空间的选取器
    def add_target_word_embeddings_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            with tf.variable_scope("target_word"):
                _word_embeddings = tf.get_variable('embedding', shape=[self.ntarwords, self.args.embedding_size],
                                                   initializer=self.initializer, regularizer=self.l2_regularizer)
                word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.target_word_ids)

            if self.args.use_pretrain_target:
                pre_train_size = self.target_embedding.shape[0]
                self.target_embedding_init = _word_embeddings[:pre_train_size].assign(self.target_embedding)

            self.target_word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    # 添加模型组件
    def add_logits_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            # 源域lstm模型
            with tf.variable_scope('src_lstm'):
                # lstm_hidden -- LSTM网络单元的个数，即隐藏层的节点数
                # forward, backward
                cell_fw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)
                cell_bw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)

                # 双向lstm模型
                # sequence_length = Used to copy-through state and zero-out outputs when past a batch element's sequence length.
                # 大小为[batch_size],数据的类型是int32/int64向量。如果当前时间步的index超过该序列的实际长度时，
                # 则该时间步不进行计算，RNN的state复制上一个时间步的，同时该时间步的输出全部为零。
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.src_word_embeddings, sequence_length=self.src_sequence_lengths,
                    dtype=tf.float32)
                outout = tf.concat([output_fw, output_bw], axis=-1)

                # tf.nn.dropout() 是tf里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
                # Dropout就是在不同的训练过程中随机扔掉一部分神经元。
                # 也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，
                # 也不参加神经网络的计算。但是它的权重得保留下来，因为下次样本输入时它可能又得工作了
                outout = tf.nn.dropout(outout, self.dropout)
                # batch_size * ntime_steps * (2 self.args.lstm_hidden)
                self.src_after_specific = outout

            # 目标域lstm模型
            with tf.variable_scope('target_lstm'):
                cell_fw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)
                cell_bw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.target_word_embeddings, sequence_length=self.target_sequence_lengths,
                    dtype=tf.float32)
                outout = tf.concat([output_fw, output_bw], axis=-1)
                outout = tf.nn.dropout(outout, self.dropout)
                self.target_after_specific = outout

            # 线性转换，用于将源域output转换成标签的概率
            with tf.variable_scope('src_lstm_linear'):
                W = tf.get_variable("W", shape=[2 * self.args.lstm_hidden, self.ntags],
                                    dtype=tf.float32, initializer=self.initializer, regularizer=self.l2_regularizer)
                b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                    initializer=self.initializer, regularizer=self.l2_regularizer)
                # 获取输出的第二维 timestep 的大小
                ntime_steps = tf.shape(self.src_after_specific)[1]
                # 将输出转换成二维矩阵，每一行是单个字符的输出结果
                output = tf.reshape(self.src_after_specific, [-1, 2 * self.args.lstm_hidden])
                # 将转化好的矩阵进行线性转换
                pred = tf.matmul(output, W) + b
                # 将预测结果再转回去输出的形式，batchSize * ntimeSteps * ntags
                self.src_logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

            # 线性转换，用于将目标域output转换成标签的概率
            with tf.variable_scope('target_lstm_linear'):
                W = tf.get_variable("W", shape=[2 * self.args.lstm_hidden, self.ntags],
                                    dtype=tf.float32, initializer=self.initializer, regularizer=self.l2_regularizer)
                b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                    initializer=self.initializer, regularizer=self.l2_regularizer)
                ntime_steps = tf.shape(self.target_after_specific)[1]
                output = tf.reshape(self.target_after_specific, [-1, 2 * self.args.lstm_hidden])
                pred = tf.matmul(output, W) + b
                self.target_logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])
                tf.add_to_collection('target_logits', self.target_logits)

    def add_loss_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            # CRF loss
            with tf.variable_scope('src_crf'):
                self.src_log_likelihood, self.src_transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.src_logits, self.labels, self.src_sequence_lengths
                )
            with tf.variable_scope('target_crf'):
                self.target_log_likelihood, self.target_transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.target_logits, self.labels, self.target_sequence_lengths
                )
            tf.add_to_collection('target_transition_params', self.target_transition_params)
            # 这里是每次都将一个 batch 20条句子丢进去训练
            # 但是在计算似然损失时，只算各自域的句子的损失
            self.src_crf_loss = tf.reduce_mean(-self.src_log_likelihood[:self.src_batch_size])
            self.target_crf_loss = tf.reduce_mean(-self.target_log_likelihood[self.src_batch_size:])

            # Adaptive loss
            if self.args.penalty_ratio > 0:
                # src_after_specific --> [batch_size * ntime_steps * (2 self.args.lstm_hidden)]
                # 将有效字符向量全部挑出来，放在一个一个数组当中
                self.src_depad = _de_pad(self.src_after_specific, self.src_sequence_lengths)
                self.target_depad = _de_pad(self.target_after_specific, self.target_sequence_lengths)

                # 将分别在两个模型中训练的20条句子的有效字符词向量放在一维数组当中
                # 选择对应的算法，计算差别(损失)，达到用源域对目标域进行修正的效果
                if self.args.penalty == 'mmd':
                    with tf.name_scope('mmd'):
                        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5,
                                  1e6]
                        gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
                        loss_value = MMD(self.src_depad, self.target_depad, kernel=gaussian_kernel)
                        mmd_loss = tf.maximum(1e-4, loss_value)

                    self.penalty_loss = self.args.penalty_ratio * mmd_loss
                elif self.args.penalty == 'kl':
                    self.src_depad_sm = tf.nn.softmax(self.src_depad)
                    self.target_depad_sm = tf.nn.softmax(self.target_depad)
                    self.kl_loss = MKL(self.src_depad_sm, self.target_depad_sm)
                    self.penalty_loss = self.args.penalty_ratio * self.kl_loss
                elif self.args.penalty == 'cmd':
                    self.cmd_loss = CMD(self.src_depad, self.target_depad, 5)
                    self.penalty_loss = self.args.penalty_ratio * self.cmd_loss
                else:
                    self.logger.critical("Penalty Type Invalid.")
                    exit(9)
            # 若惩罚损失参数大于零，加上 penalty_loss
                temp = self.src_crf_loss + self.target_crf_loss + self.penalty_loss
            else:
            # 否则损失暂为两个模型分别各自训练的损失
                temp = self.src_crf_loss + self.target_crf_loss

            # 正则化损失
            if self.args.use_l2:
                self.l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                temp1 = temp + self.l2_loss
            else:
                temp1 = temp

            # 如果不是共享crf，则要计算两个转换参数的差距（损失）
            if not self.args.share_crf:
                self.crf_l2_loss = tf.nn.l2_loss(
                    self.target_transition_params - self.src_transition_params) * self.args.crf_l2_ratio
                temp2 = temp1 + self.crf_l2_loss
            else:
                temp2 = temp1
            # 总的损失
            self.loss = temp2

    def add_train_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            with tf.variable_scope('train'):
                if self.args.optim.lower() == 'adam':
                    optimizer = tf.train.AdamOptimizer(self.lr)
                elif self.args.optim.lower() == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(self.lr)
                else:
                    raise NotImplementedError("Unknown optim {}".format(self.args.optim))
                # 训练目标即为最小化损失
                self.train_op = optimizer.minimize(self.loss)

    def add_init_op(self):
        # 初始化器
        self.init = tf.global_variables_initializer()

    def build(self):
        self.add_placeholder()  # 加入各占位符
        self.add_src_word_embeddings_op()   # 加入存储源域词向量的变量
        if not self.args.share_embed:
            self.add_target_word_embeddings_op()    # 加入存储目标域词向量的变量
        self.add_logits_op()    # 定义模型组件
        self.add_loss_op()  # 定义损失函数
        self.add_train_op() # 定义训练方式
        self.add_init_op()  # 定义初始化方式
        self.logger.info("Model info: {}".format(self.describe))

    def preBuild(self):
        graph = tf.get_default_graph()
        self.src_word_ids = graph.get_operation_by_name('src_word_ids').outputs[0]
        self.src_sequence_lengths = graph.get_operation_by_name('src_sequence_lengths').outputs[0]
        self.target_word_ids = graph.get_operation_by_name('target_word_ids').outputs[0]
        self.target_sequence_lengths = graph.get_operation_by_name('target_sequence_lengths').outputs[0]
        self.sequence_lengths = graph.get_operation_by_name('sequence_lengths').outputs[0]
        self.batch_size = graph.get_operation_by_name('batch_size').outputs[0]
        self.is_training = graph.get_operation_by_name('is_training').outputs[0]
        self.labels = graph.get_operation_by_name('labels').outputs[0]
        self.dropout = graph.get_operation_by_name('dropout').outputs[0]
        self.lr = graph.get_operation_by_name('lr').outputs[0]
        self.target_logits = graph.get_collection('target_logits')[0]
        self.target_transition_params = graph.get_collection('target_transition_params')[0]


    def predict_batch(self, sess, words=None, target_words=None, mode='target', is_training=True):
        feed_dict, sequence_lengths = self.get_feed_dict(words, None, target_words=target_words, dropout=1.0, mode=mode,
                                                         is_training=is_training)
        viterbi_sequences = []
        logits, transition_params = sess.run([self.target_logits, self.target_transition_params],
                                             feed_dict=feed_dict)
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, transition_params
            )
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences, sequence_lengths

    def run_epoch(self, sess, src_train, src_dev, tags, target_train, target_dev, n_epoch_noimprove):
        # batch 的数量
        nbatches = (len(target_train) + self.target_batch_size - 1) // self.target_batch_size
        # 进度条
        prog = Progbar(target=nbatches)
        total_loss = 0

        # src, target 每次返回三个数组words,tags,target_words，数组长度为batch_size
        src = minibatches(src_train, self.src_batch_size, circle=True)
        target = minibatches(target_train, self.target_batch_size, circle=True)

        # 一次迭代
        for i in range(nbatches):
            src_words, src_tags, _ = next(src)
            target_words, target_tags, _ = next(target)
            # 总真实标签
            labels = src_tags + target_tags

            feed_dict, _ = self.get_feed_dict(src_words, labels, target_words, self.args.learning_rate,
                                              self.args.dropout, self.src_batch_size, is_training=True)

            if self.args.penalty_ratio > 0:
                _, src_crf_loss, target_crf_loss, penalty_loss, loss = sess.run(
                    [self.train_op, self.src_crf_loss, self.target_crf_loss, self.penalty_loss, self.loss],
                    feed_dict=feed_dict)
                try:
                    prog.update(i + 1,
                                [("train loss", loss[0]), ("src crf", src_crf_loss), ("target crf", target_crf_loss),
                                 ("{} loss".format(self.args.penalty), penalty_loss)])
                except:
                    prog.update(i + 1,
                                [("train loss", loss), ("src crf", src_crf_loss), ("target crf", target_crf_loss),
                                 ("{} loss".format(self.args.penalty), penalty_loss)])
            else:
                _, src_crf_loss, target_crf_loss, loss = sess.run(
                    [self.train_op, self.src_crf_loss, self.target_crf_loss, self.loss],
                    feed_dict=feed_dict)
                try:
                    prog.update(i + 1,
                                [("train loss", loss[0]), ("src crf", src_crf_loss), ("target crf", target_crf_loss)])
                except:
                    prog.update(i + 1,
                                [("train loss", loss), ("src crf", src_crf_loss), ("target crf", target_crf_loss)])
            total_loss += loss

        # 迭代后
        self.info['loss'] += [total_loss / nbatches]
        acc, p, r, f1 = self.run_evaluate(sess, target_train, tags, target='target')
        self.info['dev'].append((acc, p, r, f1))
        self.logger.critical(
            "target train acc {:04.2f}  f1  {:04.2f}  p {:04.2f}  r  {:04.2f}".format(100 * acc, 100 * f1, 100 * p,
                                                                                      100 * r))
        acc, p, r, f1 = self.run_evaluate(sess, target_dev, tags, target='target')
        self.info['dev'].append((acc, p, r, f1))
        self.logger.info(
            "dev acc {:04.2f}  f1  {:04.2f}  p {:04.2f}  r  {:04.2f}".format(100 * acc, 100 * f1, 100 * p, 100 * r))
        return acc, p, r, f1

    def run_evaluate(self, sess, test, tags, target='src'):
        # tags 是标签字典
        # test 测试集
        accs = []
        # accs : [True, True, False, False, False, False, True]
        correct_preds, total_correct, total_preds = 0., 0., 0.
        nbatces = (len(test) + self.args.batch_size - 1) // self.args.batch_size
        prog = Progbar(target=nbatces)
        for i, (words, labels, target_words) in enumerate(minibatches(test, self.args.batch_size)):
            # 得到预测的标签序列
            if target == 'src':
                labels_pred, sequence_lengths = self.predict_batch(sess, words, mode=target, is_training=False)
            else:
                labels_pred, sequence_lengths = self.predict_batch(sess, None, words, mode=target, is_training=False)

            for lab, label_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = label_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                # get_chunks(sequence, tags)
                # tags = {'B': 0, 'E': 1, 'S': 2, 'I': 3}
                # seq = [0,3,3,1,2,2]
                # 输出：[('B', 0, 1), ('I', 1, 3), ('E', 3, 4), ('S', 4, 5), ('S', 5, 6)]
                lab_chunks = set(get_chunks(lab, tags))
                # set : {('I', 1, 3), ('E', 3, 4), ('B', 0, 1), ('S', 4, 5), ('S', 5, 6)}
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

            prog.update(i + 1)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, p, r, f1

    def predict(self, sess, test, id_to_tag, id_to_word):
        nbatces = (len(test) + self.args.batch_size - 1) // self.args.batch_size
        prog = Progbar(target=nbatces)
        with open(self.args.predict_out, 'w+', encoding='utf8') as outfile:
            for i, (words, target_words, true_words) in enumerate(minibatches_evaluate(test, self.args.batch_size)):
                labels_pred, sequence_lengths = self.predict_batch(sess, target_words=words)

                for word, true_word, label_pred, length in zip(words, true_words, labels_pred, sequence_lengths):
                    true_word = true_word[:length]
                    lab_pred = label_pred[:length]

                    for item, tag in zip(true_word, lab_pred):
                        outfile.write(item + '\t' + id_to_tag[tag] + '\n')
                    outfile.write('\n')

                prog.update(i + 1)

    def train(self, src_train, src_dev, tags, target_train, target_dev, src_batch_size, target_batch_size):
        # src_train = Dataset(args.train_file, processing_word, processing_tag, None)
        # src_dev = Dataset(args.dev_file, processing_word, processing_tag, None)
        # tags ：标签字典
        # target_train = Dataset(args.target_train_file, processing_target_word, processing_tag)
        # target_dev = Dataset(args.target_dev_file, processing_target_word, processing_tag)
        best_score = -1e-4
        tf_config = tf.ConfigProto()
        # 动态申请显存
        tf_config.gpu_options.allow_growth = True
        # 限制GPU的使用率
        tf_config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_frac
        # 自动选择运行设备
        tf_config.allow_soft_placement = True

        with tf.Session(config=tf_config) as sess:
            sess.run(self.init)
            if self.args.use_pretrain_src:
                sess.run(self.src_embedding_init)
            if self.args.use_pretrain_target and self.args.flag == 1:
                sess.run(self.target_embedding_init)

            nepoch_no_imprv = 0
            for epoch in range(self.args.epoch):
                self.logger.info("Epoch : {}/{}".format(epoch + 1, self.args.epoch))

                acc, p, r, f1 = \
                    self.run_epoch(sess, src_train, src_dev, tags, target_train, target_dev, nepoch_no_imprv)
                # 返回target模型在测试集的表现

                self.args.learning_rate *= self.args.lr_decay

                if f1 > best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.args.model_output):
                        os.makedirs(self.args.model_output)
                    saver = tf.train.Saver()
                    saver.save(sess, self.args.model_output)
                    best_score = f1
                    self.logger.info("New best score: {}".format(f1))
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.args.nepoch_no_imprv:
                        self.logger.info("Early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        break

        return self.evaluate(target_dev, tags, target='target')

    def evaluate(self, test, tags, target='src'):
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_frac
        tf_config.allow_soft_placement = True
        with tf.Session(config=tf_config) as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.args.model_output)
            acc, p, r, f1 = self.run_evaluate(sess, test, tags, target=target)
            self.info['test'] = (acc, p, r, f1)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        return acc, p, r, f1
