from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras_bert import load_trained_model_from_checkpoint, AdamWarmup
import tensorflow as tf

embedding_size = 768
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'

global graph
graph = tf.get_default_graph()


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
        ps1 = Dense(1, use_bias=False)(x)
        ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
        ps2 = Dense(1, use_bias=False)(x)
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
