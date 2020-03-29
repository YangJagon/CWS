# encoding=utf-8
# Project: transfer_cws
# Author: xingjunjie
# Create Time: 07/11/2017 2:35 PM on PyCharm

import argparse
from data_utils import load_pre_train, load_vocab, get_processing, Dataset, EvaluateSet
import tensorflow as tf
import os
import pickle
import json
from utils import get_logger


def train_pos(args):
    src_embedding = None
    target_embedding = None
    logger = get_logger(args.log)
    logger.info('Model Type: {}'.format(args.type))
    # 如果config参数输入了，则在二进制文件中读取数据
    if os.path.exists(args.config) and (not args.config == 'debug.json'):
        logger.info('Loading config from {}'.format(args.config))
        config = json.load(open(args.config, 'r'))
        try:
            vocab_word = pickle.load(open(config['word'], 'rb'))
            vocab_tag = pickle.load(open(config['tag'], 'rb'))
            target_vocab_word = pickle.load(open(config['target_word'], 'rb'))

            assert len(vocab_word) == config['nword']
            assert len(vocab_tag) == config['ntag']
            assert len(target_vocab_word) == config['ntarword']

            if args.use_pretrain_src:
                _, src_embedding = load_pre_train(args.src_embedding)

            if args.use_pretrain_target:
                _, target_embedding = load_pre_train(args.target_embedding)

        except Exception as e:
            logger.error(e)
            exit(0)
    # 从字符文件中读取数据
    else:
        if args.use_pretrain_src:
            pre_dictionary, src_embedding = load_pre_train(args.src_embedding)
            vocab_word, vocab_tag = load_vocab(args.train_file, pre_dictionary)
        else:
            # 令每一个字符对应一个整型数字
            vocab_word, vocab_tag = load_vocab(args.train_file)

        if args.use_pretrain_target:
            pre_dictionary, target_embedding = load_pre_train(args.target_embedding)
            target_vocab_word, _ = load_vocab(args.train_file, pre_dictionary)
        else:
            target_vocab_word, _ = load_vocab(args.target_train_file)

        i = 0
        while os.path.exists('./.cache/vocab_{}.pickle'.format(str(i))) or os.path.exists(
                './.cache/tag_{}.pickle'.format(str(i))):
            i += 1
        if not os.path.exists('./.cache'):
            os.makedirs('./.cache')
        with open('./.cache/vocab_{}.pickle'.format(str(i)), 'wb') as vocab, open(
                './.cache/tag_{}.pickle'.format(str(i)), 'wb') as tag, open(
            './.cache/target_vocab_{}.pickle'.format(str(i)), 'wb') as tar_vocab:
            pickle.dump(vocab_word, vocab)
            pickle.dump(vocab_tag, tag)
            pickle.dump(target_vocab_word, tar_vocab)

        with open(args.config, 'w+') as config:
            json.dump({
                'word': './.cache/vocab_{}.pickle'.format(str(i)),
                'tag': './.cache/tag_{}.pickle'.format(str(i)),
                'target_word': './.cache/target_vocab_{}.pickle'.format(str(i)),
                'nword': len(vocab_word),
                'ntag': len(vocab_tag),
                'ntarword': len(target_vocab_word)
            }, config, indent='\t')

    # 字符库的大小
    nword = len(vocab_word)
    ntag = len(vocab_tag)
    ntarword = len(target_vocab_word)

    logger.info("Src:    {}  {}".format(nword, ntag))
    logger.info("Target: {}".format(ntarword))
    logger.info("Flag:   {}".format(args.flag))
    logger.info("Src embed trainable: {}".format(not args.disable_src_embed_training))
    logger.info("\ntrain:{}\ndev  :{}\ntest :{}\n\n".format(args.train_file, args.dev_file, args.test_file))
    logger.info("\nTarget: \ntrain:{}\ndev  :{}\ntest :{}\n".format(args.target_train_file, args.target_dev_file,
                                                                    args.target_test_file))
    logger.info("MSG:   {}\n".format(args.msg))
    logger.info("lr_ratio: {}\n".format(str(args.lr_ratio)))
    logger.info("penalty_ratio: {}\n".format(str(args.penalty_ratio)))
    logger.info("penalty: {}\n".format(str(args.penalty)))

    # 返回一个函数，该函数接受一个字符参数，返回对应的整型数字
    processing_word = get_processing(vocab_word)
    processing_tag = get_processing(vocab_tag)
    processing_target_word = get_processing(target_vocab_word)

    # 返回一个数据集，该数据集用于读取文件
    # 遍历该数据集时，每次读取一句话并返回三个数组:words,tags,target_words
    # 该数据集的长度即为该文件句子的数量
    src_train = Dataset(args.train_file, processing_word, processing_tag, None)
    src_dev = Dataset(args.dev_file, processing_word, processing_tag, None)
    src_test = Dataset(args.test_file, processing_word, processing_tag, None)

    target_train = Dataset(args.target_train_file, processing_target_word, processing_tag)
    target_dev = Dataset(args.target_dev_file, processing_target_word, processing_tag)
    target_test = Dataset(args.target_test_file, processing_target_word, processing_tag)

    src_len = len(src_train)
    target_len = len(target_train)
    ratio = target_len / (src_len + target_len)
    logger.info("\nsrc:    {}\ntarget: {}\n".format(src_len, target_len))

    # src_batch_size = args.batch_size - target_batch_size
    # 因训练时是一个batch一个batch的数据丢进去训练的
    # ratio = 0.1 if ratio < 0.1 else ratio
    target_batch_size = int(ratio * args.batch_size)
    target_batch_size = 1 if target_batch_size < 1 else target_batch_size
    src_batch_size = args.batch_size - target_batch_size
    logger.info("\nsrc_batch_size: {}\ntarget_batch_size: {}".format(src_batch_size, target_batch_size))
    assert target_batch_size >= 0

    model = Model(args, ntag, nword, ntarwords=ntarword, src_embedding=src_embedding, target_embedding=target_embedding,
                  logger=logger, src_batch_size=src_batch_size)

    model.build()
    try:
        if args.debug:
            model.train(src_dev, src_dev, vocab_tag, target_dev, target_dev, src_batch_size, target_batch_size)
        else:
            model.train(src_train, src_dev, vocab_tag, target_train, target_dev, src_batch_size, target_batch_size)
            # src_train = Dataset(args.train_file, processing_word, processing_tag, None)
            # src_dev = Dataset(args.dev_file, processing_word, processing_tag, None)
            # target_train = Dataset(args.target_train_file, processing_target_word, processing_tag)
            # target_dev = Dataset(args.target_dev_file, processing_target_word, processing_tag)
            # vocab_tag ：标签字典

    except KeyboardInterrupt:
        model.evaluate(target_dev, vocab_tag, target='target')


def predict(args):
    config = json.load(open(args.config, 'r'))
    try:
        vocab_word = pickle.load(open(config['word'], 'rb'))
        vocab_tag = pickle.load(open(config['tag'], 'rb'))
        target_vocab_word = pickle.load(open(config['target_word'], 'rb'))

        assert len(vocab_word) == config['nword']
        assert len(vocab_tag) == config['ntag']
        assert len(target_vocab_word) == config['ntarword']
    except Exception as e:
        print(e)
        exit(0)

    # id_to_word = {value: key for key, value in vocab_word.items()}
    id_to_word = {value: key for key, value in target_vocab_word.items()}
    id_to_tag = {value: key for key, value in vocab_tag.items()}
    # processing_word = get_processing(vocab_word)
    processing_word = get_processing(target_vocab_word)
    predict = EvaluateSet(args.predict_file, processing_word)

    # 自加
    logger = get_logger(args.log)
    # target_batch_size = 1
    # src_batch_size = args.batch_size - target_batch_size

    # model = Model(args, len(vocab_tag), len(vocab_word), ntarwords=len(target_vocab_word))
    model = Model(args, len(vocab_tag), len(vocab_word), ntarwords=len(target_vocab_word), logger=logger)
    # model = Model(args, ntag, nword, ntarwords=ntarword, src_embedding=src_embedding,
    # target_embedding=target_embedding, logger=logger, src_batch_size=src_batch_size)
    # model.build()

    saver = tf.train.import_meta_graph(model.args.model_input + '.meta')
    # saver = tf.train.Saver()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = True
    tf_config.allow_soft_placement = True
    tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    # tf_config.gpu_options.per_process_gpu_memory_fraction = model.args.gpu_frac
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, model.args.model_input)
        model.preBuild()
        # saver.restore(sess, "./model/checkpoint")
        model.predict(sess, predict, id_to_tag, id_to_word)

    print('result saved in {}'.format(args.predict_out))


def main(args):
    if args.func == 'train':
        train_pos(args)
    elif args.func == 'predict':
        predict(args)


if __name__ == '__main__':
    """
    Functions
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=['train', 'predict'], help='Function to run.')

    """
    Several paths
    """
    parser.add_argument('--log', type=str, default="./debug.log", help="path to log file")
    parser.add_argument('--src_embedding', type=str, help="Path to pretrained embedding.")
    parser.add_argument('--target_embedding', type=str, help="Path to pretrained embedding.")

    """
    Model type
    """
    parser.add_argument('-t', '--type', type=str, default='1', choices=['1', '2', '3'], help="Model type")

    """
    Shared Hyper parameters
    """
    parser.add_argument('--batch_size', type=int, default=20, help="Training batch size")
    parser.add_argument('--epoch', type=int, default=100, help="Training epoch")
    parser.add_argument('--optim', type=str, default='Adam', help="optimizer, SGD or Adam")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.99, help="Learning rate decay rate")
    parser.add_argument('--embedding_size', type=int, default=50, help="Embedding size")
    # 词向量的长度

    """
    training
    """
    # lstm模型输出的向量长度
    parser.add_argument('--lstm_hidden', type=int, default=50, help="Hidden dimension of lstm model.")
    parser.add_argument('--dropout', type=float, default=0.8, help="Dropout rate of lstm.")

    parser.add_argument('--model_output', type=str, default='./model/debug')
    parser.add_argument('--model_input', type=str, default='./model/debug', help='path of model used for predict')

    parser.add_argument('--train_file', type=str, default='./data/pku_train.txt')
    parser.add_argument('--dev_file', type=str, default='./data/pku_dev.txt')
    parser.add_argument('--test_file', type=str, default='./data/pku_dev.txt')

    parser.add_argument('--target_train_file', type=str, default='medical_data/forum_train.txt')
    parser.add_argument('--target_dev_file', type=str, default='medical_data/forum_dev.txt')
    parser.add_argument('--target_test_file', type=str, default='medical_data/forum_test.txt')

    parser.add_argument('--use_pretrain_src', action="store_true")
    parser.add_argument('--use_pretrain_target', action="store_true")
    parser.add_argument('--nepoch_no_imprv', type=int, default=5, help="Num of epoch with no improvement")
    parser.add_argument('--gpu_frac', type=float, default=1.0)
    parser.add_argument('-d', '--debug', action='store_true', help='Flag for debug.')

    parser.add_argument('--config', type=str, default='debug.json', help='Path to saved config file')

    parser.add_argument('--flag', type=int, default=0, help='training flag')
    parser.add_argument('--disable_src_embed_training', action="store_true", default=False)
    parser.add_argument('--msg', default='No msg.')
    parser.add_argument('--matrix', default='matrix.p')
    parser.add_argument('--use_adapt', action="store_true")
    parser.add_argument('--lr_ratio', default=1.0, type=float)
    parser.add_argument('--gpu_device', default=0, type=int)
    parser.add_argument('--share_crf', action="store_true")
    parser.add_argument('--share_embed', action="store_true")
    parser.add_argument('--use_l2', action="store_true")
    parser.add_argument('--l2_ratio', default=0.1, type=float)
    parser.add_argument('--crf_l2_ratio', default=0.3, type=float)
    parser.add_argument('-p', '--penalty', type=str, default='mmd', choices=['kl', 'mmd', 'cmd'])
    parser.add_argument('--penalty_ratio', default=0.05, type=float)

    """
    Predict
    """
    parser.add_argument('--predict_file', type=str, help='Path to file for prediction')
    parser.add_argument('--predict_out', type=str, default='predict_out.txt', help='Path to save predict result.')

    args = parser.parse_args()

    global Model
    Model = getattr(__import__('model_{}'.format(args.type)), 'Model')
    # 导入model_(i).py模块中的Model，并返回一个模型实例
    main(args)
