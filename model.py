# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import logging

from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

from collections import namedtuple

GRU4Rec_HParams = namedtuple('GRU4Rec_HParams',
                             'reset_after_session, layers, rnn_size, loss, final_act, hidden_act,'
                             'dropout_p_hidden, batch_size, optimizer, learning_rate, decay, decay_steps, sigma,'
                             'grad_cap, init_as_normal, n_sample, sample_alpha, smoothing, n_epochs')

Session_HParams = namedtuple('Session_HParams',
                             'n_items, session_key, item_key, time_key, train_random_order, time_sort')


class GRU4Rec:

    def __init__(self, args, session_args, sess, mode):
        self.logger = logging.getLogger('main.model')
        self.sess = sess
        self.is_training = mode

        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.grad_cap = args.grad_cap
        self.n_sample = args.n_sample
        self.sample_alpha = args.sample_alpha
        self.smoothing = args.smoothing
        self.optimizer = args.optimizer

        self.session_key = session_args.session_key
        self.item_key = session_args.item_key
        self.time_key = session_args.time_key
        self.train_random_order = session_args.train_random_order
        self.time_sort = session_args.time_sort
        self.n_items = session_args.n_items

        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.final_act == 'linear':
            self.final_activation = self.linear
        elif args.final_act == 'relu':
            self.final_activatin = self.relu
        else:
            self.final_activation = self.tanh

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            self.loss_function = self.top1
        elif args.loss.startswith('bpr-max-'):
            self.loss_function = self.bpr_max
            self.bpreg = float(args.loss[8:])
        elif args.loss == 'top1-max':
            self.loss_function = self.top1_max
        elif args.loss == 'xe_logit':
            self.loss_function = self.cross_entropy_logits
        else:
            raise NotImplementedError

        self.build_model()

        # use self.predict_state to hold hidden states during prediction.
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def relu(self, X):
        return tf.nn.relu(X)

    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))

    def softmax_neg(self, X):
        hack_matrix = np.ones((self.batch_size, self.batch_size + self.n_sample), dtype=np.float32)
        np.fill_diagonal(hack_matrix, 0)
        self.hack_matrix = tf.Variable(hack_matrix, trainable=False)

        if hasattr(self, 'hack_matrix'):
            X = X * self.hack_matrix
            e_x = tf.exp(X - tf.reduce_max(X, axis=1, keep_dims=True)) * self.hack_matrix
        else:
            e_x = tf.matrix_set_diag(tf.exp(X - tf.reduce_max(X, axis=1, keep_dims=True)),
                                     tf.zeros(shape=(self.batch_size)))

        return e_x / tf.reduce_sum(e_x, axis=1, keep_dims=True)

    ############################LOSS FUNCTIONS######################
    def cross_entropy(self, yhat):
        if self.smoothing:
            n_out = self.batch_size + self.n_sample
            term1 = (1.0 - (n_out / (n_out - 1)) * self.smoothing) * (-tf.log(tf.diag_part(yhat) + 1e-24))
            term2 = (self.smoothing / (n_out - 1)) * tf.reduce_sum(-tf.log(yhat + 1e-24), axis=1)
            return tf.reduce_mean(term1 + term2)
        else:
            return tf.reduce_mean(-tf.log(tf.diag_part(yhat) + 1e-24))

    def cross_entropy_logits(self, yhat):
        if self.smoothing:
            n_out = self.batch_size + self.n_sample
            term1 = (1.0 - (n_out / (n_out - 1)) * self.smoothing) * tf.diag_part(yhat)
            term2 = (self.smoothing / (n_out - 1)) * tf.reduce_sum(yhat, axis=1)
            return tf.reduce_mean(term1 + term2)
        else:
            return tf.reduce_mean(tf.diag_part(yhat))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def bpr_max(self, yhat):
        yhatT = tf.transpose(yhat)
        softmax_scores = tf.transpose(self.softmax_neg(yhat))
        term1 = -tf.log(tf.reduce_sum(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT) * softmax_scores, axis=0) + 1e-24)
        term2 = self.bpreg * tf.reduce_sum((yhatT**2) * softmax_scores, axis=0)
        return tf.reduce_mean(term1 + term2)

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT**2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def top1_max(self, yhat):
        yhatT = tf.transpose(yhat)
        softmax_scores = tf.transpose(self.softmax_neg(yhat))
        y = softmax_scores * (tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT**2))
        return tf.reduce_mean(tf.reduce_sum(y, axis=0))

    def build_model(self):

        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size + self.n_sample], name='output')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in range(self.layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

            cell = GRUCell(self.rnn_size, activation=self.hidden_act,
                           kernel_initializer=initializer, bias_initializer=tf.constant_initializer(0.0))
            drop_cell = DropoutWrapper(cell, output_keep_prob=self.dropout)
            stacked_cell = MultiRNNCell([drop_cell] * self.layers)

            inputs = tf.nn.embedding_lookup(embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state

        '''
        Use other examples of the minibatch as negative samples.
        '''
        sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
        sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
        logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
        self.yhat = self.final_activation(logits)
        self.cost = self.loss_function(self.yhat)

        valid_logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
        self.valid_yhat = self.final_activation(valid_logits)

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))

        '''
        Try different optimizers. 
        '''
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.9, epsilon=1e-6)
        if self.optimizer == 'rmsprop':
          optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer == 'momentum':
          optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.25)
        elif self.optimizer == 'adagrad':
          optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def init(self, data):
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return offset_sessions

    def generate_neg_samples(self, pop, length):
        if self.sample_alpha:
            sample = np.searchsorted(pop, np.random.rand(self.n_sample * length))
        else:
            sample = np.random.choice(self.n_items, size=self.n_sample * length)
        if length > 1:
            sample = sample.reshape((length, self.n_sample))
        return sample

    def fit(self, data, valid, sample_store=10000000):
        train_data = data

        import time

        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}), on=self.item_key, how='inner')
        offset_sessions = self.init(data)
        base_order = np.argsort(data.groupby(self.session_key)[self.time_key].min().values) if self.time_sort else np.arange(len(offset_sessions) - 1)

        if self.n_sample:
            pop = data.groupby('ItemId').size()
            pop = pop[self.itemidmap.index.values].values**self.sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            if sample_store:
                generate_length = sample_store // self.n_sample
                if generate_length <= 1:
                    sample_store = 0
                    self.logger.debug('No example store was used')
                else:
                    neg_samples = self.generate_neg_samples(pop, generate_length)
                    sample_pointer = 0
                    self.logger.debug('Created sample store with {} batches of samples'.format(generate_length))
            else:
                self.logger.debug('No example store was used')

        for epoch in range(self.n_epochs):
            t1 = time.time()

            epoch_cost = []
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]

            session_idx_arr = np.random.permutation(len(offset_sessions) - 1) if self.train_random_order else base_order
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]
            finished = False
            while not finished:
                minlen = (end - start).min()
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start + i + 1]

                    if self.n_sample:
                        if sample_store:
                            if sample_pointer == generate_length:
                                neg_samples = self.generate_neg_samples(pop, generate_length)
                                sample_pointer = 0
                            sample = neg_samples[sample_pointer]
                            sample_pointer += 1
                        else:
                            sample = self.generate_neg_samples(pop, 1)
                        y = np.hstack([out_idx, sample])
                    else:
                        y = out_idx

                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: y, self.dropout: self.dropout_p_hidden}
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        self.logger.error(str(epoch) + ':Nan error!')
                        return
                    #if step == 1 or step % self.decay_steps == 0:
                    #    avgc = np.mean(epoch_cost)
                    #    t2 = time.time()
                    #    self.logger.info('epoch:%2s, step:%5d, lr:%.6f, loss:%.6f, elpased(s):%6.3f',epoch, step, lr, avgc, t2-t1)
                    #    #print('Elpased: %.4fs' % (t2 - t1))

                start = start + minlen - 1
                mask = np.arange(len(iters))[(end - start) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions) - 1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter] + 1]
                if len(mask) and self.reset_after_session:
                    for i in range(self.layers):
                        state[i][mask] = 0

            avgc = np.mean(epoch_cost)

            if np.isnan(avgc):
                self.logger.error('Epoch {}: Nan error!'.format(epoch, avgc))
                return
            #saver.save(self.sess, '{}/gru-model'.format(checkpoint_dir), global_step=epoch)
            
            if epoch % 2 == 0:
              res = self.evaluate_sessions_batch(train_data, valid, batch_size=self.batch_size)
              self.logger.info('epoch:%2d, loss:%.6f, recall@20:%.10f, mrr@20:%.10f',epoch, avgc, res[0], res[1])



    def evaluate_sessions_batch(self, train_data, test_data, batch_size, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'):
        '''
        Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    
        Parameters
        --------
        train_data : It contains the transactions of the train set. In evaluation phrase, this is used to build item-to-id map.
        test_data : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        cut-off : int
            Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
        batch_size : int
            Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
        session_key : string
            Header of the session ID column in the input file (default: 'SessionId')
        item_key : string
            Header of the item ID column in the input file (default: 'ItemId')
        time_key : string
            Header of the timestamp column in the input file (default: 'Time')
    
        Returns
        --------
        out : tuple
            (Recall@N, MRR@N)
    
        '''
        self.predict = False
        # Build itemidmap from train data.
        itemids = train_data[item_key].unique()
        itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
    
        test_data.sort_values([session_key, time_key], inplace=True)
        offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
        evalutation_point_count = 0
        mrr, recall = 0.0, 0.0
        if len(offset_sessions) - 1 < batch_size:
            batch_size = len(offset_sessions) - 1
        iters = np.arange(batch_size).astype(np.int32)
        maxiter = iters.max()
        start = offset_sessions[iters]
        end = offset_sessions[iters + 1]
        in_idx = np.zeros(batch_size, dtype=np.int32)
        np.random.seed(42)
    
        while True:
            valid_mask = iters >= 0
            if valid_mask.sum() == 0:
                break
            start_valid = start[valid_mask]
            minlen = (end[valid_mask] - start_valid).min()
            in_idx[valid_mask] = test_data[item_key].values[start_valid]
            for i in range(minlen - 1):
                out_idx = test_data[item_key].values[start_valid + i + 1]
                preds = self.predict_next_batch(iters, in_idx, itemidmap, batch=batch_size)
                preds.fillna(0, inplace=True)
                in_idx[valid_mask] = out_idx
                ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
                rank_ok = ranks < cut_off
                recall += rank_ok.sum()
                mrr += (1.0 / ranks[rank_ok]).sum()
                evalutation_point_count += len(ranks)
            start = start + minlen - 1
            mask = np.arange(len(iters))[(valid_mask) & (end - start <= 1)]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(offset_sessions) - 1:
                    iters[idx] = -1
                else:
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[maxiter]
                    end[idx] = offset_sessions[maxiter + 1]
    
        return recall / evalutation_point_count, mrr / evalutation_point_count


    def predict_next_batch(self, session_ids, input_item_ids, itemidmap, batch):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1
            self.predict = True

        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:  # change internal states with session changes
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session = session_ids.copy()

        in_idxs = itemidmap[input_item_ids]
        fetches = [self.valid_yhat, self.final_state]
        feed_dict = {self.X: in_idxs, self.dropout: 1.0}
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T
        return pd.DataFrame(data=preds, index=itemidmap.index)

