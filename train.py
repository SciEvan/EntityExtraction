# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 训练文件
import numpy as np
import tensorflow as tf

from config import *
from entity_extraction import DealText, NERModel

dt = DealText(TRAIN_DATA)
words, labels = dt.reader_text()
word_sequence = dt.get_sequence(words, name='words')
label_sequence = dt.get_sequence(labels, name='labels')
word_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=word_sequence,
                                                              maxlen=int(MAX_WORD_LENGTH),
                                                              padding='post',
                                                              dtype=np.int32,
                                                              truncating='post')

label_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=label_sequence,
                                                               maxlen=int(MAX_WORD_LENGTH),
                                                               padding='post',
                                                               dtype=np.int32,
                                                               truncating='post')
tensor_slices = tf.data.Dataset.from_tensor_slices((word_sequence, label_sequence))
model = NERModel()

optimizers = tf.keras.optimizers.Adam(learning_rate=float(LEARNING_RATE))
for epoch in range(int(EPOCHS)):
    dataset = tensor_slices.shuffle(buffer_size=int(BUFFER_SIZE)).batch(batch_size=int(BATCH_SIZE))
    for index, (word, label) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred, text_lens, log_likelihood = model.call(word=word, label=label)
            loss = - tf.reduce_mean(log_likelihood)
        grads = tape.gradient(target=loss, sources=model.trainable_variables)
        optimizers.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        if index % 20 == 0:
            print(epoch, loss)

tf.saved_model.save(model, SAVE_MODEL_DIR, signatures={'call': model.call})
