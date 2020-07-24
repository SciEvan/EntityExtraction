# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:18
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 构建模型文件
import os

import tensorflow as tf
import tensorflow_addons as tfa

from config import *


class NERModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        tag_size = int(os.environ['TAG_SIZE']) + 1
        self.embedding = tf.keras.layers.Embedding(int(os.environ['VOCAB_SIZE']) + 1, output_dim=int(EMBEDDING_SIZE))
        # self.cnn = tf.keras.layers.Conv1D(kernel_size=5, strides=2, padding='same', filters=32)
        # self.dense = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)
        # self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.bi_lst = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True))
        self.dense1 = tf.keras.layers.Dense(units=tag_size, activation=tf.keras.activations.softmax)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.params = tf.Variable(tf.random.uniform(shape=(tag_size, tag_size)))

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, int(MAX_WORD_LENGTH)), dtype=tf.int32, name='word'),
                                  tf.TensorSpec(shape=(None, int(MAX_WORD_LENGTH)), dtype=tf.int32, name='label')])
    def call(self, word, label):
        text_length = tf.math.reduce_sum(tf.cast(tf.math.not_equal(word, 0), dtype=tf.int32), axis=1)
        x = self.embedding(word)
        # x = self.cnn(x)
        # x = self.dense(x)
        # x = self.dropout(x)
        x = self.bi_lst(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        log_likelihood, self.params = tfa.text.crf_log_likelihood(inputs=x,
                                                                  tag_indices=label,
                                                                  sequence_lengths=text_length,
                                                                  transition_params=self.params)
        return x, text_length, log_likelihood
