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
import keras.backend as K

global graph
graph = tf.get_default_graph()

max_len = 200

category_nums = 21
seed = 10

token_dict = {}
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
with open(dict_path, encoding='utf8') as file:
    for line in file.readlines():
        token = line.strip()
        token_dict[token] = len(token_dict)

additional_chars = set()


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
    random_order = shuffle(train_data, seed=seed)[0].tolist()
    train_data = random_order[0:int(0.98 * len(random_order))]
    dev_data = random_order[int(0.98 * len(random_order)):]

    # 新数据
    new_data = pd.read_csv('new_data.csv')
    for t, c, n in new_data.values:
        train_data.append((t, c, n))

    train_data = shuffle(train_data, seed=seed)[0].tolist()

    for d in train_data + dev_data:
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

    additional_chars.remove(u'，')

    # 把验证集作为测试集
    dev = pd.read_csv('data/eval.csv', encoding='utf-8', header=None)
    test_data = []
    for id, t, c in zip(dev[0], dev[1], dev[2]):
        test_data.append((id, t, c))

    return train_data, dev_data, test_data, id2class, class2id


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def data_generator(data, batch_size):
    while True:
        X, segment, start, end, max_length = [], [], [], [], 0
        for i, d in enumerate(data):
            text, c = d[0][:max_len], d[1]
            # x = f'___{c}___{text}'
            # tokens = tokenizer.tokenize(first=text, second=c)
            # if len(tokens) > max_length:
            #     max_length = len(tokens)

            sub = d[2]
            # sub_token = tokenizer.tokenize(sub)[1:-1]
            # s = list_find(tokens, sub_token)
            s = text.find(sub)
            if s != -1:
                e = s + len(sub) - 1

                x, seg = tokenizer.encode(first=text, second=c)
                if len(x) > max_length:
                    max_length = len(x)

                X.append(x)
                segment.append(seg)
                start.append(s)
                end.append(e)

                if len(X) == batch_size or i == len(data) - 1:
                    X = pad_sequences(X, maxlen=max_length)
                    segment = pad_sequences(segment, maxlen=max_length)
                    start = one_hot(start, max_length)
                    end = one_hot(end, max_length)
                    yield [X, segment, start, end], None
                    X, segment, start, end, max_length = [], [], [], [], 0


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


new_data = []


def extract_entity(text, category, class2id, model):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if category not in class2id.keys():
        return 'NaN'

    # text_in = u'___%s___%s' % (category, text)
    # text_in = text_in[:510]
    # _tokens = tokenizer.tokenize(text_in)
    # _tokens = tokenizer.tokenize(first=text, second=category)
    text = text[:400]
    x, s = tokenizer.encode(first=text, second=category, max_len=512)
    prob_s, prob_e = model.predict([np.array([x]), np.array([s])])
    prob_s, prob_e = softmax(prob_s[0]), softmax(prob_e[0])

    for i, t in enumerate(text):
        if len(t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t) and t not in additional_chars:
            prob_s[i] -= 10
    start = prob_s.argmax()

    for end in range(start, len(text)):
        t = text[end]
        if len(t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t) and t not in additional_chars:
            break
    end = prob_e[start:end + 1].argmax() + start
    res = ''.join(text[start: end + 1])

    if prob_s[start] > 0.9 and prob_e[end] > 0.9:
        new_data.append([text, category, res])

    return res


class Evaluate(Callback):
    def __init__(self, data, model, test_model, class2id):
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.dev_data = data
        self.model = model
        self.test_model = test_model
        self.class2id = class2id

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc >= self.best:
            self.best = acc
            self.model.save_weights('output/subject_model.weights')
        print('epoch: %d, acc: %.4f, best acc: %.4f\n' % (epoch, acc, self.best))

    # def evaluate(self):
    #     eps = 0
    #     error_list = []
    #     for d in tqdm(iter(self.dev_data)):
    #         R = extract_entity(d[0], d[1], self.class2id, self.test_model)
    #         if R == d[2]:
    #             eps += 1
    #         else:
    #             error_list.append((d[0], d[1], d[2], R))
    #     with open('error.txt', 'w', encoding='utf-8')as file:
    #         file.write(str(error_list))
    #     return eps / len(self.dev_data)

    def evaluate(self):
        eps = 0
        for d in tqdm(iter(self.dev_data)):
            R = extract_entity(d[0], d[1], self.class2id, self.test_model)
            if R == d[2]:
                eps += 1
        pre = eps / len(self.dev_data)

        return 2 * pre / (pre + 1)


def dev(dev_data, class2id, test_model):
    eps = 0
    error_list = []
    for d in tqdm(iter(dev_data)):
        R = extract_entity(d[0], d[1], class2id, test_model)
        if R == d[2]:
            eps += 1
        else:
            error_list.append((d[0], d[1], d[2], R))
    with open('error.txt', 'w', encoding='utf-8')as file:
        file.write(str(error_list))

    pre = eps / len(dev_data)
    return (2 * pre) / (1 + pre)


def test(test_data, class2id, test_model):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    with open('result.txt', 'w', encoding='utf-8')as file:
        for d in tqdm(iter(test_data)):
            s = str(d[0]) + ',' + extract_entity(d[1].replace('\t', ''), d[2], class2id, test_model)
            file.write(s + '\n')

    print('length: ', len(new_data))
    import json
    dic = {'data': new_data}
    with open('new_data.txt', 'w', encoding='utf-8')as file:
        file.write(json.dumps(dic, ensure_ascii=False))


if __name__ == '__main__':
    batch_size = 16
    learning_rate = 1e-3
    min_learning_rate = 1e-5
    epochs = 100
    is_test = False

    train_data, dev_data, test_data, id2class, class2id = read_data()

    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_data),
        batch_size=batch_size,
        epochs=epochs,
        warmup_proportion=0.1,
    )

    model, test_model = Graph(total_steps, warmup_steps, lr=learning_rate, min_lr=min_learning_rate)

    if is_test:
        test_model.load_weights('output/subject_model.weights')
        model.load_weights('output/subject_model.weights')
        test(test_data, class2id, test_model)
        # acc = dev(dev_data, class2id, test_model)
        # print('acc: ', acc)
    else:
        # test_model.load_weights('output/subject_model.weights')
        # model.load_weights('output/subject_model.weights')

        evaluator = Evaluate(dev_data, model, test_model, class2id)
        X = data_generator(train_data, batch_size)
        steps = int((len(train_data) + batch_size - 1) / batch_size)

        model.fit_generator(X, steps_per_epoch=100, epochs=epochs, callbacks=[evaluator])
