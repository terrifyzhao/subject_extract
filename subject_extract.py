#! -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from keras_layer_normalization import LayerNormalization
from keras.layers import *
from data_utils import *

mode = 0
maxlen = 128
learning_rate = 1e-4
min_learning_rate = 1e-5

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

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

D = pd.read_csv('data/train.csv', encoding='utf-8', header=None)
D = D[D[2] != u'其他']
classes = set(D[2].unique())

train_data = []
for t, c, n in zip(D[1], D[2], D[3]):
    train_data.append((t, c, n))

if not os.path.exists('data/random_order_train.json'):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('data/random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('data/random_order_train.json'))

dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
# 新数据
# new_data = pd.read_csv('new_data.csv')
# for t, c, n in new_data.values:
#     train_data.append((t, c, n))
# shuffle(train_data, seed=0)[0].tolist()

additional_chars = set()
for d in train_data + dev_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

additional_chars.remove(u'，')

D = pd.read_csv('data/eval.csv', encoding='utf-8', header=None)
test_data = []
for id, t, c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        return None

    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        super(Attention, self).build(input_shape)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs, **kwargs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, c = d[0][:maxlen], d[1]
                text = u'___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[2]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                    s1[start] = 1
                    s2[end] = 1
                    x1, x2 = tokenizer.encode(first=text)
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1, X2, S1, S2], None
                        X1, X2, S1, S2 = [], [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))  # 待识别句子输入
x2_in = Input(shape=(None,))  # 待识别句子输入
s1_in = Input(shape=(None,))  # 实体左边界（标签）
s2_in = Input(shape=(None,))  # 实体右边界（标签）

x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

x = bert_model([x1, x2])

# x_s = Attention(16, 48)([x, x, x, x_mask, x_mask])
# x_s = Lambda(lambda x: x[0] + x[1])([x, x_s])
# x_s = LayerNormalization()(x_s)
# x_s_co = Dense(768, use_bias=False)(x_s)
# x_s_out = Lambda(lambda x: x[0] + x[1])([x_s, x_s_co])
# x_s_out = LayerNormalization()(x_s_out)
# x_s_out = Lambda(lambda x: x[0] * x[1])([x_s_out, x_mask])
# ps1 = Dense(1, use_bias=False)(x_s_out)
# ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
#
# x_e = Attention(16, 48)([x, x, x, x_mask, x_mask])
# x_e = Lambda(lambda x: x[0] + x[1])([x, x_e])
# x_e = LayerNormalization()(x_e)
# x_e_co = Dense(768, use_bias=False)(x_e)
# x_e_out = Lambda(lambda x: x[0] + x[1])([x_e, x_e_co])
# x_e_out = LayerNormalization()(x_e_out)
# x_e_out = Lambda(lambda x: x[0] * x[1])([x_e_out, x_mask])
# ps2 = Dense(1, use_bias=False)(x_e_out)
# ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

ps1 = Dense(1, use_bias=False)(x)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
ps2 = Dense(1, use_bias=False)(x)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

model = Model([x1_in, x2_in], [ps1, ps2])

train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
loss = loss1 + loss2

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


new_data_list = []


def extract_entity(text_in, c_in):
    if c_in not in classes:
        return 'NaN'
    text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
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

    if _ps1[start] > 0.9 and _ps2[end] > 0.9:
        new_data_list.append([text_in, c_in, a])

    return a


class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.
        self.passed = 0

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
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        eps = 0
        error_list = []
        for d in tqdm(iter(dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                eps += 1
            else:
                error_list.append((d[0], d[1], d[2], R))
        with open('error.txt', 'w', encoding='utf-8')as file:
            file.write(str(error_list))
        return eps / len(dev_data)


def test():
    with open('result.txt', 'w', encoding='utf-8')as file:
        for d in tqdm(iter(test_data)):
            s = str(d[0]) + ',' + extract_entity(d[1].replace('\t', ''), d[2])
            file.write(s + '\n')

    print('length: ', len(new_data_list))
    import json
    dic = {'data': new_data_list}
    with open('new_data.txt', 'w', encoding='utf-8')as file:
        file.write(json.dumps(dic, ensure_ascii=False))


evaluator = Evaluate()
train_D = data_generator(train_data)

if __name__ == '__main__':

    isTest = False

    if isTest:
        model.load_weights('best_model.weights')
        test()
    else:
        # model.load_weights('best_model.weights')
        # train_model.load_weights('best_model.weights')
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=100,
                                  callbacks=[evaluator]
                                  )
else:
    train_model.load_weights('best_model.weights')
