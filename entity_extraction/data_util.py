# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:19
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 基础的数据处理
import json
import os

from tensorflow.keras.preprocessing import text

from config import *


class DealText:

    def __init__(self, text_path):
        self.text_path = text_path
        self.dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

    def reader_text(self):
        """读取文件"""
        words, labels = [], []
        with open(file=self.text_path, mode='r', encoding=ENCODING)as fp:
            for line in fp:
                if TEXT_SPLIT_SEP not in line:
                    continue
                word, label = line.strip().split('\t')
                words.append(word)
                labels.append(label)
        return words, labels

    def get_sequence(self, words: list, name):
        """获取words的序列

        输入一维列表，如： words: ['我,和，他','他,和,你']
                                    分割符配置 TEXT_SPLIT_SEP=',' \n
                                    最大值配置 MAX_UNM_WORD=3000 \n
                                    文档编码 ENCODING='utf8'

                       name: 列表会解析成文件保存。保存的名字也name为参数。

        :param words: 输入一个列表,['AB','BC'],列表中的每一句话都是用 TEXT_SPLIT_SEP 符号给分割，参考：config配置文件
        :param name: 加载自定义映射表
        :return: 该字符的序列
        """
        token = text.Tokenizer(num_words=MAX_UNM_WORD, split=TEXT_SPLIT_SEP, filters=TEXT_SPLIT_SEP)
        index_word_file = os.path.join(self.dir, f'index_{name}.json')
        word_index_file = os.path.join(self.dir, f'{name}_index.json')
        if not (os.path.exists(index_word_file) and os.path.exists(word_index_file)):
            token.fit_on_texts(words)
            with open(index_word_file, mode='w', encoding=ENCODING)as wf:
                json.dump(token.index_word, fp=wf, ensure_ascii=False, indent=4)
            with open(word_index_file, mode='w', encoding=ENCODING)as wf:
                json.dump(token.word_index, fp=wf, ensure_ascii=False, indent=4)
        else:
            with open(index_word_file, mode='r', encoding=ENCODING) as rf:
                index_word = json.load(rf)
                token.index_word = index_word
            with open(word_index_file, mode='r', encoding=ENCODING) as rf:
                word_index = json.load(rf)
                token.word_index = word_index
        sequence = token.texts_to_sequences(words)
        return sequence
