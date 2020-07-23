# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2020/7/23 下午2:21
# @Author: 张伟
# @EMAIL: Jtyoui@qq.com
# @Notes : 配置文件
from os import environ

MAX_UNM_WORD = environ.get('MAX_UNM_WORD', 3000)
TEXT_SPLIT_SEP = environ.get('TEXT_SPLIT_SEP', '\x02')
ENCODING = environ.get('ENCODING', 'utf8')
