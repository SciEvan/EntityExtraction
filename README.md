# EntityExtraction
TensorFlow2+Python3.8实现实体抽取模型

# 下载数据
    https://github.com/jtyoui/EntityExtraction/releases/download/v1.0/data.zip

# 下载训练好的模型
    https://github.com/jtyoui/EntityExtraction/releases/download/v1.0/models.zip

## 使用方法
    # 克隆代码到指定目录 /mnt/EntityExtraction
    git clone https://github.com/jtyoui/EntityExtraction.git /mnt/EntityExtraction
    
    # 切换到代码目录
    cd /mnt/EntityExtraction
    
    # 下载数据
    wget https://github.com/jtyoui/EntityExtraction/releases/download/v1.0/data.zip
    
    # 解压数据,解压数据到 /mnt/EntityExtraction/data
    zip data.zip
    
    # 安装第三方库
    pip install -r requirements.txt -i https://pypi.douban.com/simple
    
    # 训练数据
    python3 ./train.py
    
    # 测试数据并生成评估文件
    python3 ./assess.py
    
    # 执行评估脚本读取评估文件，生成评估结果
    python3 ./CoNLL-2000.py ./data/assess.tsv
    

## 参数脚本
```python
# config.py
from os import environ
MAX_UNM_CHAR = environ.setdefault('MAX_UNM_CHAR', '3000')
TEXT_SPLIT_SEP = environ.setdefault('TEXT_SPLIT_SEP', '\x02')
ENCODING = environ.setdefault('ENCODING', 'utf8')
VOCAB_SIZE = environ.setdefault('VOCAB_SIZE', 'AUTO')
EMBEDDING_SIZE = environ.setdefault('EMBEDDING_SIZE', '250')
TAG_SIZE = environ.setdefault('TAG_SIZE', 'AUTO')
BUFFER_SIZE = environ.setdefault('BUFFER_SIZE', '1000')
BATCH_SIZE = environ.setdefault('BATCH_SIZE', '32')
EPOCHS = environ.setdefault('EPOCHS', '10')
MAX_WORD_LENGTH = environ.setdefault('MAX_WORD_LENGTH', '200')
LEARNING_RATE = environ.setdefault('LEARNING_RATE', '0.001')
SAVE_MODEL_DIR = environ.setdefault('SAVE_MODEL_DIR', 'models/1')
TRAIN_DATA = environ.setdefault('TRAIN_DATA', 'data/train.tsv')
TEST_DATA = environ.setdefault('TEST_DATA', 'data/test.tsv')
```

## 评估结果
```text
processed 107866 tokens with 3699 phrases; found: 3525 phrases; correct: 2856.
accuracy:  97.42%; precision:  81.02%; recall:  77.21%; FB1:  79.07
              LOC: precision:  84.92%; recall:  78.67%; FB1:  81.68  
              ORG: precision:  75.77%; recall:  74.77%; FB1:  75.27  
              PER: precision:  79.23%; recall:  76.90%; FB1:  78.05  
```

## 模型保存文件夹格式
```text
/ner        # 保存文件夹取的名字
    /1      # 版本号为1的模型文件
        /assets
        /variables
        saved_model.pb
    ...
    /N      # 版本号为N的模型文件
        /assets
        /variables
        saved_model.pb
```

## 模型部署
    docker pull tensorflow/serving
    docker run -d -p 8501:8501 -v /mnt/ner:/models/ner -e MODEL_NAME=ner --name=tensorflow-serving tensorflow/serving
    
## REStFul接口调用
```python
import requests
import json
import os

dirs = os.path.dirname(__file__)
index_tag = json.load(open(os.path.join(dirs, 'data', 'index_labels.json'), mode='r', encoding='utf-8'))
tag_index = {tag: int(index) for index, tag in index_tag.items()}
index_word = json.load(open(os.path.join(dirs, 'data', 'index_words.json'), mode='r', encoding='utf-8'))
word_index = {word: int(index) for index, word in index_word.items()}

url = 'http://IP:8501/v1/models/ner:predict'
headers = {"content-type": "application/json"}

def ner(words: str):
    length = len(words)
    assert length <= 200, ValueError('输入的字符串最大长度为200')
    word = [word_index.get(i, 0) for i in words]
    data = json.dumps({
        'inputs': {
            "word": [word + [0] * (200 - length)],
            'label': [[0] * 200],
        },
        'signature_name': 'call'
    })
    result = requests.post(url, data=data, headers=headers).json()
    value = result['outputs']['output_0']
    ls = [v.index(max(v)) for v in value[0]]

    print(ls[:length])
    print(words)

if __name__ == '__main__':
    ner('我叫李伟')
```