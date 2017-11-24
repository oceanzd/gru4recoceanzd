# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import gru4rec

## 全数据集
#PATH_TO_TRAIN = '/home/ljm/data/yoochoose/preprocessed/click_only/full/rsc15_train_full.txt'
#PATH_TO_TEST = '/home/ljm/data/yoochoose/preprocessed/click_only/full/rsc15_test.txt'

# 十分之一数据集
PATH_TO_TRAIN = '/home/ljm/dataset/yoochoose/sampled-one-tenth/rsc15_train_tr.txt'
PATH_TO_TEST = '/home/ljm/dataset/yoochoose/sampled-one-tenth/rsc15_train_valid.txt'


# 日志模块
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('th-GRU4Rec.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# 命令行参数
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("rnn_size", 100, "rnn_size")
flags.DEFINE_string("loss", "top1", "loss")
flags.DEFINE_string("final_act", "tanh", "final_act")
flags.DEFINE_string("hidden_act", "tanh", "hidden_act")
flags.DEFINE_float("dropout_p_hidden", 0.5, "dropout_p_hidden")
flags.DEFINE_integer("batch_size", 50, "batch_size")
flags.DEFINE_integer("n_epochs", 3, "n_epochs")
flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
flags.DEFINE_string("optimizer", "adam", "optimizer")

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
    
    #Reproducing results from "Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    
    #adapt : None, 'adagrad', 'rmsprop', 'adam', 'adadelta'
    logger.info('rnn_size:%3d  batch_size:%s  hidden_act:%s  dropout_p_hidden:%s  final_act:%s  optimizer:%s  learning_rate:%s', FLAGS.rnn_size, FLAGS.batch_size, FLAGS.hidden_act, FLAGS.dropout_p_hidden, FLAGS.final_act, FLAGS.optimizer, FLAGS.learning_rate)
    gru = gru4rec.GRU4Rec(loss=FLAGS.loss, final_act=FLAGS.final_act, hidden_act=FLAGS.hidden_act, layers=[FLAGS.rnn_size], batch_size=FLAGS.batch_size, dropout_p_hidden=FLAGS.dropout_p_hidden, learning_rate=FLAGS.learning_rate, momentum=0.0, time_sort=False, n_epochs=FLAGS.n_epochs, adapt=FLAGS.optimizer)
    gru.fit(data, valid)
    
    #Reproducing results from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    
    #print('Training GRU4Rec with 100 hidden units')

    #gru = gru4rec.GRU4Rec(loss='bpr-max-0.5', final_act='linear', hidden_act='tanh', layers=[100], batch_size=32, dropout_p_hidden=0.0, learning_rate=0.2, momentum=0.5, n_sample=2048, sample_alpha=0, time_sort=True)
    #gru.fit(data)
    #
    #res = evaluation.evaluate_sessions_batch(gru, valid, None)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
