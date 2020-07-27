# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 生成评估文件
import json
import os

import tensorflow as tf
import tensorflow_addons as tfa

from config import *
from entity_extraction import DealText

dt = DealText(TEST_DATA)
words, labels = dt.reader_text()
word_sequence = dt.get_sequence(words, name='words')
label_sequence = dt.get_sequence(labels, name='labels')
index_tag = json.load(open(os.path.join(dt.dir, 'index_labels.json'), mode='r', encoding=ENCODING))
index_word = json.load(open(os.path.join(dt.dir, 'index_words.json'), mode='r', encoding=ENCODING))
index_tag = {int(k): v.upper() for k, v in index_tag.items()}

word_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=word_sequence,
                                                              maxlen=int(MAX_WORD_LENGTH),
                                                              padding='post',
                                                              dtype='int32',
                                                              truncating='post')

label_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=label_sequence,
                                                               maxlen=int(MAX_WORD_LENGTH),
                                                               padding='post',
                                                               dtype='int32',
                                                               truncating='post')
tensor_slices = tf.data.Dataset.from_tensor_slices((word_sequence, label_sequence))
dataset = tensor_slices.shuffle(buffer_size=int(BUFFER_SIZE)).batch(batch_size=int(BATCH_SIZE))

model = tf.saved_model.load(SAVE_MODEL_DIR)
wf = open(os.path.join(dt.dir, 'assess.tsv'), mode='w', encoding=ENCODING)
for words, labels in dataset:
    y_pred, text_lens, _ = model.call(word=words, label=labels)
    for index, (y, l) in enumerate(zip(y_pred, text_lens)):
        word = [index_word[str(i)] for i in words[index][:l].numpy()]
        label = [index_tag[i] for i in labels[index][:l].numpy()]
        pred_label = tfa.text.viterbi_decode(y[:l], model.params)[0]
        pred_label = [index_tag[i] for i in pred_label]
        for w, p, r in zip(word, label, pred_label):
            wf.write(w + '\t' + p + '\t' + r + '\n')
    wf.flush()
wf.close()
