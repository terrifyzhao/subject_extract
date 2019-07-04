from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

embedding_size = 128


class Attention(Layer):
    """多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

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

    def call(self, inputs):
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


def graph(id2char, class2id, learning_rate):
    x_in = Input(shape=(None,))  # 待识别句子输入
    c_in = Input(shape=(1,))  # 事件类型
    start_in = Input(shape=(None,))  # 实体左边界（标签）
    end_in = Input(shape=(None,))  # 实体右边界（标签）

    x, c, start, end = x_in, c_in, start_in, end_in
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)

    x = Embedding(len(id2char) + 2, embedding_size)(x)
    c = Embedding(len(class2id), embedding_size)(c)
    c = Lambda(lambda x: x[0] * 0 + x[1])([x, c])

    x = Add()([x, c])
    x = Dropout(0.2)(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = Bidirectional(CuDNNLSTM(embedding_size // 2, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
    x = Bidirectional(CuDNNLSTM(embedding_size // 2, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    xo = x
    x = Attention(8, 16)([x, x, x, x_mask, x_mask])
    x = Lambda(lambda x: x[0] + x[1])([xo, x])

    x = Concatenate()([x, c])

    x1 = Dense(embedding_size, use_bias=False, activation='tanh')(x)
    ps1 = Dense(1, use_bias=False)(x1)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])

    x2 = Dense(embedding_size, use_bias=False, activation='tanh')(x)
    ps2 = Dense(1, use_bias=False)(x2)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

    test_model = Model([x_in, c_in], [ps1, ps2])

    train_model = Model([x_in, c_in, start_in, end_in], [ps1, ps2])

    loss1 = K.mean(K.categorical_crossentropy(start_in, ps1, from_logits=True))
    loss2 = K.mean(K.categorical_crossentropy(end_in, ps2, from_logits=True))
    loss = loss1 + loss2

    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(learning_rate))
    train_model.summary()

    return train_model, test_model
