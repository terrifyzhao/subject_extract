from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras_bert import load_trained_model_from_checkpoint, AdamWarmup
import tensorflow as tf
from keras_layer_normalization import LayerNormalization

embedding_size = 768
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'

global graph
graph = tf.get_default_graph()


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


def Graph(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5):
    with graph.as_default():
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        x_in = Input(shape=(None,))
        c_in = Input(shape=(None,))
        start_in = Input(shape=(None,))
        end_in = Input(shape=(None,))

        x, c, start, end = x_in, c_in, start_in, end_in
        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)

        x = bert_model([x, c])

        x_s = Attention(16, 48)([x, x, x, x_mask, x_mask])
        x_s = Lambda(lambda x: x[0] + x[1])([x, x_s])
        x_s = LayerNormalization()(x_s)
        x_s_co = Dense(768, use_bias=False)(x_s)
        x_s_out = Lambda(lambda x: x[0] + x[1])([x_s, x_s_co])
        x_s_out = LayerNormalization()(x_s_out)
        x_s_out = Lambda(lambda x: x[0] * x[1])([x_s_out, x_mask])
        ps1 = Dense(1, use_bias=False)(x_s_out)
        ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])

        x_e = Attention(16, 48)([x, x, x, x_mask, x_mask])
        x_e = Lambda(lambda x: x[0] + x[1])([x, x_e])
        x_e = LayerNormalization()(x_e)
        x_e_co = Dense(768, use_bias=False)(x_e)
        x_e_out = Lambda(lambda x: x[0] + x[1])([x_e, x_e_co])
        x_e_out = LayerNormalization()(x_e_out)
        x_e_out = Lambda(lambda x: x[0] * x[1])([x_e_out, x_mask])
        ps2 = Dense(1, use_bias=False)(x_e_out)
        ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

        test_model = Model([x_in, c_in], [ps1, ps2])

        train_model = Model([x_in, c_in, start_in, end_in], [ps1, ps2])

        loss1 = K.mean(K.categorical_crossentropy(start_in, ps1, from_logits=True))
        ps2 -= (1 - K.cumsum(start, 1)) * 1e10
        loss2 = K.mean(K.categorical_crossentropy(end_in, ps2, from_logits=True))
        loss = loss1 + loss2

        train_model.add_loss(loss)
        train_model.compile(optimizer=AdamWarmup(total_steps, warmup_steps, lr, min_lr))
        train_model.summary()

        return train_model, test_model
