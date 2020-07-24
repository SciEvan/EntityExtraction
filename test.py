# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 测试文件
import tensorflow as tf

from config import SAVE_MODEL_DIR, MAX_WORD_LENGTH
from entity_extraction import DealText

if __name__ == '__main__':
    dt = DealText('/home/jtyoui/Downloads/病理实体数据/test.tsv')
    words, labels = dt.reader_text()
    word_sequence = dt.get_sequence(words[:10], name='words')
    label_sequence = dt.get_sequence(labels[:10], name='labels')
    word_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences=word_sequence,
                                                                  maxlen=int(MAX_WORD_LENGTH),
                                                                  padding='post',
                                                                  dtype='int32',
                                                                  truncating='post')
    label = tf.constant(value=[[0] * int(MAX_WORD_LENGTH)], dtype=tf.int32)
    model = tf.saved_model.load(SAVE_MODEL_DIR)
    for i in word_sequence:
        word = tf.expand_dims(i, axis=0)
        y_pred, _, _ = model.call(word=word, label=label)
        print(y_pred)
        break
