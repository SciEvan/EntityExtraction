# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 测试文件
from entity_extraction import DealText

if __name__ == '__main__':
    dt = DealText('/home/jtyoui/Downloads/病理实体数据/train.tsv')
    words, labels = dt.reader_text()
    word_sequence = dt.get_sequence(words[:10], name='words')
    print(word_sequence)
    label_sequence = dt.get_sequence(labels[:10], name='labels')
    print(label_sequence)
