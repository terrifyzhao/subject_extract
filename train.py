import json
from tqdm import tqdm
import os
import pandas as pd
from data_utils import *
from keras.callbacks import Callback
from model import Graph
from keras_bert import Tokenizer, calc_train_steps
import tensorflow as tf

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


def data_generator(data, batch_size, class2id):
    while True:
        X, category, start, end = [], [], [], []
        for i, d in enumerate(data):
            ques = d[0]
            cat = d[1]
            sub = d[2]
            s = ques.find(sub)
            if s != -1:
                e = s + len(sub) - 1
                ques = tokenizer.tokenize(ques)
                c = class2id.get(cat)
                cat = tokenizer.tokenize(cat)
                x, _ = tokenizer.encode(first=cat, second=ques)

                X.append(x)
                category.append(c)
                start.append(s)
                end.append(e)

                if len(X) == batch_size or i == len(data) - 1:
                    X = pad_sequences(X, maxlen=max_len)
                    category = one_hot(category, category_nums)
                    start = one_hot(start, max_len)
                    end = one_hot(end, max_len)
                    yield [X, category, start, end], None
                    X, category, start, end = [], [], [], []


def extract_entity(text, category, class2id, model):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if category not in class2id.keys():
        return 'NaN'
    text = tokenizer.tokenize(text)
    category = tokenizer.tokenize(category)
    x, _ = tokenizer.encode(first=category, second=text)

    with graph.as_default():
        prob_s, prob_e = model.predict([x, one_hot(category, category_nums)])
        start = prob_s.argmax()
        end = prob_e.argmax()
        res = text[start:end + 1]
        return res

    # text_in = u'___%s___%s' % (c_in, text_in)
    # text_in = text_in[:510]
    # _tokens = tokenizer.tokenize(text_in)
    # _x1, _x2 = tokenizer.encode(first=text_in)
    # _x1, _x2 = np.array([_x1]), np.array([_x2])
    # with graph.as_default():
    #     _ps1, _ps2 = model.predict([_x1, _x2])
    #     _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    #     for i, _t in enumerate(_tokens):
    #         if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
    #             _ps1[i] -= 10
    #     start = _ps1.argmax()
    #     for end in range(start, len(_tokens)):
    #         _t = _tokens[end]
    #         if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
    #             break
    #     end = _ps2[start:end + 1].argmax() + start
    #     a = text_in[start - 1: end]
    #     return a


class Evaluate(Callback):
    def __init__(self, data, model, test_model, class2id):
        self.ACC = []
        self.best = 0.
        self.dev_data = data
        self.model = model
        self.test_model = test_model
        self.class2id = class2id

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            self.model.save_weights('output/subject_model.weights')
        print('epoch: %d, acc: %.4f, best acc: %.4f\n' % (epoch, acc, self.best))

    def evaluate(self):
        eps = 0
        for d in tqdm(iter(self.dev_data)):
            R = extract_entity(d[0], d[1], self.class2id, self.test_model)
            if R == d[2]:
                eps += 1
        return eps / len(self.dev_data)


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
    min_learning_rate = 1e-5
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

    if is_test:
        test_model.load_weights('output/subject_model.weights')
        model.load_weights('output/subject_model.weights')
        test(test_data, class2id, test_model)
    else:
        # test_model.load_weights('output/best_model.weights')
        # model.load_weights('output/best_model.weights')

        evaluator = Evaluate(dev_data, model, test_model, class2id)
        X = data_generator(train_data, batch_size, class2id)
        steps = int((len(train_data) + batch_size - 1) / batch_size)

        model.fit_generator(X, steps_per_epoch=steps, epochs=epochs, callbacks=[evaluator])
