import json
from tqdm import tqdm
import os
import pandas as pd
from data_utils import *
from keras.callbacks import Callback
from model import Graph
from keras_bert import Tokenizer
import codecs
import re
import tensorflow as tf

min_count = 2
maxlen = 256

dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
additional_chars = set()
global graph
graph = tf.get_default_graph()

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def read_data():
    # 读取数据，排除“其他”类型，其他对应的结果是nan
    D = pd.read_csv('data/train.csv', encoding='utf-8', header=None)
    D = D[D[2] != '其他']
    D = D[D[1].str.len() <= maxlen]

    # 统计所有存在的事件类型
    if not os.path.exists('data/classes.json'):
        id2class = dict(enumerate(D[2].unique()))
        class2id = {j: i for i, j in id2class.items()}
        json.dump([id2class, class2id], open('data/classes.json', 'w', encoding='utf-8'), ensure_ascii=False)
    else:
        id2class, class2id = json.load(open('data/classes.json', encoding='utf-8'))

    # 移除事件主体不在原句子中的数据
    train_data = []
    for t, c, n in zip(D[1], D[2], D[3]):
        start = t.find(n)
        if start != -1:
            train_data.append((t, c, n))

    if not os.path.exists('data/vocab.json'):
        chars = {}
        # 统计所有问题的字的个数
        for d in tqdm(iter(train_data)):
            for c in d[0]:
                chars[c] = chars.get(c, 0) + 1
        # 把出现次数少于min_count的字直接去除
        chars = {i: j for i, j in chars.items() if j >= min_count}
        id2char = {i + 2: j for i, j in enumerate(chars)}  # 0: mask, 1: padding
        char2id = {j: i for i, j in id2char.items()}
        json.dump([id2char, char2id], open('data/vocab.json', 'w', encoding='utf-8'), ensure_ascii=False)
    else:
        id2char, char2id = json.load(open('data/vocab.json', encoding='utf-8'))

    # shuffle一下并划分数据集
    random_order = shuffle(train_data)[0].tolist()
    train_data = random_order[0:int(0.9 * len(random_order))]
    dev_data = random_order[int(0.9 * len(random_order)):]

    # 把验证集作为测试集
    D = pd.read_csv('data/eval.csv', encoding='utf-8', header=None)
    test_data = []
    for id, t, c in zip(D[0], D[1], D[2]):
        test_data.append((id, t, c))

    for d in train_data + dev_data:
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

    # additional_chars.remove(u'，')

    return train_data, dev_data, test_data, id2class, class2id, id2char, char2id


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


def data_generator(data, batch_size, char2id, class2id):
    max_len = max([len(d[0]) for d in data])
    while True:
        X, category, start, end = [], [], [], []
        for d in data:
            x = '___%s___%s' % (d[1], d[0])
            tokens = tokenizer.tokenize(x)
            e = d[2]
            e_tokens = tokenizer.tokenize(e)[1:-1]

            # start, end = np.zeros(len(tokens)), np.zeros(len(tokens))
            s = list_find(tokens, e_tokens)
            if s != -1:
                e = s + len(e_tokens) - 1
                x, c = tokenizer.encode(first=x)
                X.append(x)
                category.append(c)
                start.append(s)
                end.append(e)

                if len(X) == batch_size or d == data[-1]:
                    X = pad_sequences(X, maxlen=max_len)
                    category = pad_sequences(category, max_len)
                    start = one_hot(start, max_len)
                    end = one_hot(end, max_len)
                    yield [X, np.array(category), start, end], None
                    X, category, start, end = [], [], [], []


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text_in, c_in, class2id, char2id, model):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if c_in not in class2id.keys():
        return 'NaN'
    text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    with graph.as_default():
        _ps1, _ps2 = model.predict([_x1, _x2])
        _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
        for i, _t in enumerate(_tokens):
            if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
                _ps1[i] -= 10
        start = _ps1.argmax()
        for end in range(start, len(_tokens)):
            _t = _tokens[end]
            if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
                break
        end = _ps2[start:end + 1].argmax() + start
        a = text_in[start - 1: end]
        return a
    # if c_in not in class2id:
    #     return 'NaN'
    # _x = [char2id.get(c, 1) for c in text_in]
    # _x = np.array([_x])
    # _c = np.array([[class2id[c_in]]])
    # _ps1, _ps2 = model.predict([_x, _c])
    # start = _ps1[0].argmax()
    # end = _ps2[0][start:].argmax() + start
    #
    # name = text_in[start: end + 1]
    # name.replace('(', '')
    # name.replace(')', '')
    # if len(name) > 10:
    #     name = name[0:10]
    # return name


class Evaluate(Callback):
    def __init__(self, data, model, test_model, class2id, char2id):
        self.ACC = []
        self.best = 0.
        self.dev_data = data
        self.model = model
        self.test_model = test_model
        self.class2id = class2id
        self.char2id = char2id

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            self.model.save_weights('output/best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        for d in tqdm(iter(self.dev_data)):
            R = extract_entity(d[0], d[1], self.class2id, self.char2id, self.test_model)
            if R == d[2]:
                A += 1
        return A / len(self.dev_data)


def test(test_data, class2id, char2id, test_model):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    with open('result.txt', 'w', encoding='utf-8')as file:
        for d in tqdm(iter(test_data)):
            s = str(d[0]) + ',' + extract_entity(d[1].replace('\t', ''), d[2], class2id, char2id, test_model)
            file.write(s + '\n')


if __name__ == '__main__':
    batch_size = 16
    learning_rate = 1e-4
    is_test = True

    train_data, dev_data, test_data, id2class, class2id, id2char, char2id = read_data()

    model, test_model = Graph(learning_rate=learning_rate)

    if is_test:
        test_model.load_weights('output/best_model.weights')
        model.load_weights('output/best_model.weights')
        test(test_data, class2id, char2id, test_model)
    else:
        # test_model.load_weights('output/best_model.weights')
        # model.load_weights('output/best_model.weights')

        evaluator = Evaluate(dev_data, model, test_model, class2id, char2id)

        X = data_generator(train_data, batch_size, char2id, class2id)
        steps = int((len(train_data) + batch_size - 1) / batch_size)

        model.fit_generator(X, steps_per_epoch=steps, epochs=120, callbacks=[evaluator])
