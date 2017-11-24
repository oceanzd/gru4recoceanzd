# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import logging

import model

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train", "", "PATH_TO_TRAIN")
flags.DEFINE_string("test", "", "PATH_TO_TEST")
flags.DEFINE_integer("test_model", 2, "PATH_TO_trained_model")

flags.DEFINE_integer("is_training", 0, "is_training")
flags.DEFINE_boolean("reset_after_session", True, "reset_after_session")

flags.DEFINE_integer("layers", 1, "layers")
flags.DEFINE_integer("rnn_size", 100, "rnn_size")
flags.DEFINE_string("loss", "top1", "loss")
flags.DEFINE_string("final_act", "tanh", "final_act")
flags.DEFINE_string("hidden_act", "tanh", "hidden_act")
flags.DEFINE_float("dropout_p_hidden", 0.5, "dropout_p_hidden")

flags.DEFINE_integer("batch_size", 50, "batch_size")
flags.DEFINE_integer("n_epochs", 3, "n_epochs")

flags.DEFINE_float("lr", 0.001, "learning_rate")
flags.DEFINE_string("optimizer", "adam", "optimizer")
flags.DEFINE_float("decay", 0.999, "decay")
flags.DEFINE_float("decay_steps", 1e4, "decay_steps")
flags.DEFINE_float("sigma", 0.0, "sigma")
flags.DEFINE_float("grad_cap", 0.0, "grad_cap")
flags.DEFINE_boolean("init_as_normal", False, "init_as_normal")

flags.DEFINE_integer("n_sample", 0, "n_sample")
flags.DEFINE_float("sample_alpha", 0.5, "sample_alpha")
flags.DEFINE_float("smoothing", 0.0, "smoothing")

flags.DEFINE_integer("n_items", -1, "n_items")
flags.DEFINE_string("session_key", "SessionId", "session_key")
flags.DEFINE_string("item_key", "ItemId", "item_key")
flags.DEFINE_string("time_key", "Time", "time_key")
flags.DEFINE_boolean("train_random_order", False, "train_random_order")
flags.DEFINE_boolean("time_sort", False, "time_sort")

flags.DEFINE_string("checkpoint_dir", "./checkpoint", "checkpoint_dir")

# tf logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

# logger module
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('tf-GRU4Rec.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def main(_):
    #if not os.path.exists(FLAGS.checkpoint_dir):
    #    os.mkdir(FLAGS.checkpoint_dir)

    data = pd.read_csv(FLAGS.train, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(FLAGS.test, sep='\t', dtype={'ItemId': np.int64})

    FLAGS.n_items = len(data['ItemId'].unique())

    gru4rec_hps = model.GRU4Rec_HParams(reset_after_session=FLAGS.reset_after_session, layers=FLAGS.layers,
                                        rnn_size=FLAGS.rnn_size, loss=FLAGS.loss, final_act=FLAGS.final_act,
                                        hidden_act=FLAGS.hidden_act, dropout_p_hidden=FLAGS.dropout_p_hidden,
                                        batch_size=FLAGS.batch_size, optimizer=FLAGS.optimizer, learning_rate=FLAGS.lr,
                                        decay=FLAGS.decay, decay_steps=FLAGS.decay_steps, sigma=FLAGS.sigma,
                                        grad_cap=FLAGS.grad_cap, init_as_normal=FLAGS.init_as_normal, n_sample=FLAGS.n_sample,
                                        sample_alpha=FLAGS.sample_alpha, smoothing=FLAGS.smoothing, n_epochs=FLAGS.n_epochs)

    session_hps = model.Session_HParams(n_items=FLAGS.n_items, session_key=FLAGS.session_key,
                                        item_key=FLAGS.item_key, time_key=FLAGS.time_key,
                                        train_random_order=FLAGS.train_random_order, time_sort=FLAGS.time_sort)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    logger.info('rnn_size:%3s, batch_size:%3s, lr:%6s, optimizer:%5s',
      FLAGS.rnn_size, FLAGS.batch_size, FLAGS.lr,FLAGS.optimizer)

    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(gru4rec_hps, session_hps, sess, FLAGS.is_training)
        sess.run(tf.global_variables_initializer())
        gru.fit(data, valid)

if __name__ == '__main__':
    tf.app.run()
