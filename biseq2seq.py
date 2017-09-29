# -*- coding:utf-8 -*-
import cPickle
import re
import jieba
import os
import sys
import time
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
from tensorflow.contrib import legacy_seq2seq

class Hyperparam():
    batch_size = 100
    n_epoch = 10
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5
    emb_size = 100
    state_size = 100
    num_layers = 3
    seq_length = 22
    log_dir = './out/model'
    cell_mod = 'GRU'
    gen_num = 20

class GetData():
    def __init__(self, datafiles, args):
        self.batch_size = args.batch_size
        self.data = []
        self.vocab = set()
        with open(datafiles) as f:
            for line in f:
                regex = re.compile(ur"[^\u4e00-\u9fa5a\n]")
            	line = regex.sub('', line.strip().decode('utf-8'))
            	if len(line) > 1 and u'作词' not in line and u'作曲' not in line and u'编曲' not in line:
					sen = list(jieba.cut(line))
					self.data.append(sen)
					for w in sen:
						self.vocab.add(w)
		self.totalvocab = ["PAD"] + ["UNK"] + ["GO"] + list(self.vocab)
        self.vocab_size = len(self.totalvocab)
        self.w2id = {w : i for i, w in enumerate(list(self.totalvocab))}
        self.id2w = {i : w for i, w in enumerate(list(self.totalvocab))}
        self.datanum = len(self.data) - 1
        self.x_batches = np.zeros((self.datanum, args.seq_length))
        self.y_batches = np.zeros((self.datanum, args.seq_length))
        for i in range(self.datanum):
            for j in range(len(self.data[i])):
                if i < self.datanum - 1:
                    self.x_batches[i][j] = self.w2id[self.data[i][j]]
                if i > 0:
                    self.y_batches[i - 1][j] = self.w2id[self.data[i][j]]

    def next_batch(self, index):
        # index begin with 1
        return self.x_batches[(index - 1) * self.batch_size: index * self.batch_size], self.y_batches[(index - 1) * self.batch_size : index * self.batch_size]

class Model():
    def __init__(self, args, data, isTraining = True):
		
        if not isTraining:
			args.batch_size = 1
        with tf.name_scope('data'):
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.target_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.dec_input = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        with tf.name_scope('encode'):
            if args.cell_mod == 'LSTM':
                self.cell_fw = rnn.BasicLSTMCell(args.state_size)
                self.cell_bw = rnn.BasicLSTMCell(args.state_size)
            if args.cell_mod == 'GRU':
                self.cell_fw = rnn.GRUCell(args.state_size)
                self.cell_bw = rnn.GRUCell(args.state_size)
            #self.cell_fw = rnn.MultiRNNCell([self.cell_fw] * args.num_layers)
            #self.cell_bw = rnn.MultiRNNCell([self.cell_bw] * args.num_layers)
            self.initial_state_fw = self.cell_fw.zero_state(args.batch_size, tf.float32)
            self.initial_state_bw = self.cell_bw.zero_state(args.batch_size, tf.float32)
            with tf.variable_scope('input'):
                w_in = tf.get_variable('w_in', [args.emb_size, args.state_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [data.vocab_size, args.emb_size])
                    inputs = tf.nn.embedding_lookup(tf.matmul(embedding, w_in), self.input_data)
            inputs = tf.unstack(inputs, args.seq_length, 1)
            outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(cell_fw = self.cell_fw, cell_bw = self.cell_bw, inputs = inputs, dtype = tf.float32)
            outputs_cocat = tf.stack(outputs,1)
			
        with tf.name_scope('decode'):
            with tf.variable_scope('softmax'):
                softmax_w = tf.get_variable("softmax_w", [args.state_size, data.vocab_size])
                softmax_b = tf.get_variable("softmax_b", [data.vocab_size])
                if args.cell_mod == 'LSTM':
                    self.cell = rnn.BasicLSTMCell(args.state_size)
                if args.cell_mod == 'GRU':
                    self.cell = rnn.GRUCell(args.state_size)
            def loop(prev, _):
                prev = tf.matmul(prev, softmax_w) + softmax_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(embedding, prev_symbol)
            
            decoder_inputs = tf.unstack(tf.nn.embedding_lookup(embedding, self.dec_input), args.seq_length, 1)
            dec_outputs, dec_states = legacy_seq2seq.attention_decoder(decoder_inputs, output_state_bw, outputs_cocat, self.cell,
                                                            				loop_function=loop if not isTraining else None)
        with tf.name_scope('loss'):
            output = tf.reshape(tf.stack(dec_outputs, 1), [-1, args.state_size])
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            targets = tf.reshape(self.target_data, [-1])
            loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss) / args.batch_size
            tf.summary.scalar('loss', self.cost)
        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.summary.scalar('learning_rate', self.lr)
            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.summary.histogram(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.summary.merge_all()

def train(data, model,args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
		
        max_iter = args.n_epoch * data.datanum // args.batch_size
        for i in range(max_iter):
            learning_rate = args.learning_rate * (args.decay_rate ** (i // args.decay_steps))
            x_batch, y_batch = data.next_batch( i  % (data.datanum // args.batch_size) + 1)
            #print x_batch.shape
            #print x_batch
            dec_input = np.zeros((args.batch_size, args.seq_length))
            for r in range(args.batch_size):
                for c in range(args.seq_length):
                    if c == 0:
                        dec_input[r][0] = data.w2id["GO"]
                    else:
                        dec_input[r][c] = y_batch[r][c - 1]
            feed_dict = {model.input_data: x_batch, model.target_data : y_batch, model.dec_input: dec_input, model.lr: learning_rate}
            train_loss, summary, _ = sess.run([model.cost, model.merged_op, model.train_op],feed_dict)
            if i % 10 == 0:
                print('Step:{}/{}, training_loss:{:4f}'.format(i, max_iter, train_loss))
            if i % 200 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(args.log_dir, 'lyrics_model.ckpt'), global_step=i)

def sample(data, model, args, sen):
    sen = list(jieba.cut(sen))
    inputs = np.zeros((1, args.seq_length))
    i = 0
    for w in sen:
        if i < args.seq_length:
            if w not in data.totalvocab:
                inputs[0][i] = data.w2id["UNK"]
            else:
                inputs[0][i] = data.w2id[w]
            i = i + 1
	print inputs
    dec_inputs = np.zeros((1, args.seq_length))
    dec_inputs[0][0] = data.w2id["GO"]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        lyrics = ""
        ckpt = tf.train.latest_checkpoint(args.log_dir)
        print(ckpt)
        saver.restore(sess, ckpt)
        for i in range(args.gen_num):
           # print inputs.shape
            feed_dict = {model.input_data: inputs, model.target_data: np.zeros((1, args.seq_length)), model.dec_input: dec_inputs}
            probs = sess.run([model.probs], feed_dict)
            sen = np.zeros([1, len(probs[0])])
            j = 0
            for k in probs[0]:
                w = data.id2w[np.argmax(k)]
                inputs[0][j] = np.argmax(k)
                j = j + 1
                lyrics += w
            print lyrics 
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics += '\n'
            
        return lyrics

if __name__ == '__main__':
    args = Hyperparam()
    data = GetData("./JayLyrics.txt", args)
    if len(sys.argv) == 2 and sys.argv[-1] == 1:
        isTraining = 1
    elif len(sys.argv) == 3 and sys.argv[-2] == 0:
        isTraining = 0
        sen = sys.argv[-1]
    else:
        print("参数错误")
        sys.exit(1)
    model = Model(args, data, isTraining)
    if isTraining:
        train(data, model, args)
    else:
        sample(data,model,args, sen)
		

























