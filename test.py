import json
from tqdm import tqdm
import os
import pandas as pd
from data_utils import *
from keras.callbacks import Callback
from model import Graph
from keras_bert import Tokenizer, calc_train_steps
import tensorflow as tf
import re

global graph
graph = tf.get_default_graph()

max_len = 200

category_nums = 21

token_dict = {}
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
with open(dict_path, encoding='utf8') as file:
    for line in file.readlines():
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        tokens = []
        for c in text:
            if c in self._token_dict:
                tokens.append(c)
            elif self._is_space(c):
                tokens.append('[unused1]')
            else:
                tokens.append('[UNK]')
        return tokens


tokenizer = OurTokenizer(token_dict)


def read_data():
    # 读取数据，排除“其他”类型，其他对应的结果是nan
    data = pd.read_csv('data/train.csv', header=None)
    data = data[data[2] != '其他']
    data = data[data[1].str.len() <= 256]

    # 统计所有存在的事件类型
    if not os.path.exists('data/classes.json'):
        id2class = dict(enumerate(data[2].unique()))
        class2id = {j: i for i, j in id2class.items()}
        json.dump([id2class, class2id], open('data/classes.json', 'w', encoding='utf-8'), ensure_ascii=False)
    else:
        id2class, class2id = json.load(open('data/classes.json', encoding='utf-8'))

    # 移除事件主体不在原句子中的数据
    train_data = []
    for t, c, n in zip(data[1], data[2], data[3]):
        if n in t:
            train_data.append((t, c, n))

    # shuffle一下并划分数据集
    random_order = shuffle(train_data)[0].tolist()
    train_data = random_order[0:int(0.9 * len(random_order))]
    dev_data = random_order[int(0.9 * len(random_order)):]

    # 把验证集作为测试集
    dev = pd.read_csv('data/eval.csv', encoding='utf-8', header=None)
    test_data = []
    for id, t, c in zip(dev[0], dev[1], dev[2]):
        test_data.append((id, t, c))

    return train_data, dev_data, test_data, id2class, class2id


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text, category, class2id, model):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if category not in class2id.keys():
        return 'NaN'

    text_in = f'__{category}__{text}'[0:198]
    # text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    x, s = tokenizer.encode(first=text_in)

    with graph.as_default():
        prob_s, prob_e = model.predict([np.array(x), np.array(s)])
        prob_s, prob_e = softmax(prob_s), softmax(prob_s)
        for i, _t in enumerate(_tokens):
            if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t):
                prob_s[i] -= 10
        start = prob_s.argmax()

        for end in range(start, len(_tokens)):
            _t = _tokens[end]
            if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t):
                break
        end = prob_e[start:end + 1].argmax() + start

        a = text_in[start - 1: end]
        return a


def test(test_data, class2id, test_model):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    with open('result.txt', 'w', encoding='utf-8')as file:
        for d in tqdm(iter(test_data)):
            s = str(d[0]) + ',' + extract_entity(d[1].replace('\t', ''), d[2], class2id, test_model)
            file.write(s + '\n')


if __name__ == '__main__':
    batch_size = 16
    learning_rate = 1e-3
    min_learning_rate = 1e-6
    epochs = 10
    is_test = False

    train_data, dev_data, test_data, id2class, class2id = read_data()

    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_data),
        batch_size=batch_size,
        epochs=epochs,
        warmup_proportion=0.1,
    )

    model, test_model = Graph(total_steps, warmup_steps, lr=learning_rate, min_lr=min_learning_rate)

    test_model.load_weights('output/subject_model.weights')
    model.load_weights('output/subject_model.weights')
    test(test_data, class2id, test_model)
