from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import initializers
import tensorflow as tf

class CMLA(layers.Layer):
    """
    Coupled Multi Layer Attentions.

    Adapted from the paper "Coupled Multi-Layer Attentions for Co-Extraction of
    Aspect and Opinion Terms" published in AAAI 2017.
    (http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf)
    The default values used in the parameters are the best values used in the paper.

    Parameters
    ----------
    n_tensors: int, optional, default: 20
        Number of tensors that capture the compositions between input tokens
        and the prototypes (aspect and opinion).

    n_hidden: int, optional, default: 50
        Number of hidden units.

    n_layers: int, optional, default: 2
        Number of layers for the attention network.

    dropout_rate: float, optional, default: 0.5
        Dropout rate to used during training.

    rnn_type: {'gru', 'lstm', 'bilstm', 'bigru'}, optional, default: 'gru'
        Type of RNN used.
    """

    def __init__(self, n_tensors=20, n_hidden=50, n_layers=2, dropout_rate=0.5, rnn_type='gru', **kwargs):
        super(CMLA, self).__init__(**kwargs)
        self.n_tensors = n_tensors
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.rnn_type = rnn_type

        self.dot_layer = layers.Dot(axes=0)

        if self.rnn_type == 'gru':
            self.rnn_aspect_layer = layers.GRU(units=2 * self.n_tensors,
                                               recurrent_activation='sigmoid',
                                               return_sequences=True,
                                               use_bias=False,
                                               kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                               recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                               dropout=self.dropout_rate)

            self.rnn_opinion_layer = layers.GRU(units=2 * self.n_tensors,
                                                recurrent_activation='sigmoid',
                                                return_sequences=True,
                                                use_bias=False,
                                                kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                dropout=self.dropout_rate)

        elif self.rnn_type == 'lstm':
            self.rnn_aspect_layer = layers.LSTM(units=2 * self.n_tensors,
                                                recurrent_activation='sigmoid',
                                                return_sequences=True,
                                                use_bias=False,
                                                kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                dropout=self.dropout_rate)

            self.rnn_opinion_layer = layers.LSTM(units=2 * self.n_tensors,
                                                 recurrent_activation='sigmoid',
                                                 return_sequences=True,
                                                 use_bias=False,
                                                 kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                 recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                 dropout=self.dropout_rate)

        elif self.rnn_type == 'bilstm':
            self.rnn_aspect_layer = layers.Bidirectional(layers.LSTM(units=2 * self.n_tensors,
                                                                     recurrent_activation='sigmoid',
                                                                     return_sequences=True,
                                                                     use_bias=False,
                                                                     kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                     recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                     dropout=self.dropout_rate))

            self.rnn_opinion_layer = layers.Bidirectional(layers.LSTM(units=2 * self.n_tensors,
                                                                      recurrent_activation='sigmoid',
                                                                      return_sequences=True,
                                                                      use_bias=False,
                                                                      kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                      recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                      dropout=self.dropout_rate))

        elif self.rnn_type == 'bigru':
            self.rnn_aspect_layer = layers.Bidirectional(layers.GRU(units=2 * self.n_tensors,
                                                                    recurrent_activation='sigmoid',
                                                                    return_sequences=True,
                                                                    use_bias=False,
                                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                    dropout=self.dropout_rate))

            self.rnn_opinion_layer = layers.Bidirectional(layers.GRU(units=2 * self.n_tensors,
                                                                     recurrent_activation='sigmoid',
                                                                     return_sequences=True,
                                                                     use_bias=False,
                                                                     kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                     recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                                     dropout=self.dropout_rate))
        else:
            raise ValueError("Unknown rnn type %s. Valid types are gru, lstm, bilstm, or bigru.")

    def build(self, input_shape):
        """
        Create and initialize trainable weights.
        """
        self.m0_a = self.add_weight(name='m0_a', shape=(1, self.n_hidden),
                                    initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

        self.m0_o = self.add_weight(name='m0_o', shape=(1, self.n_hidden),
                                    initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

        self.Ua = self.add_weight(name='Ua', shape=(self.n_tensors, self.n_hidden, self.n_hidden),
                                  initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

        self.Uo = self.add_weight(name='Uo', shape=(self.n_tensors, self.n_hidden, self.n_hidden),
                                  initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

        self.Va = self.add_weight(name='Va', shape=(self.n_tensors, self.n_hidden, self.n_hidden),
                                  initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

        self.Vo = self.add_weight(name='Vo', shape=(self.n_tensors, self.n_hidden, self.n_hidden),
                                  initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

        if self.n_layers > 1:
            if self.rnn_type == 'gru' or self.rnn_type == 'lstm':
                self.va = self.add_weight(name='va', shape=(2 * self.n_tensors, 1),
                                          initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

                self.vo =self.add_weight(name='vo', shape=(2 * self.n_tensors, 1),
                                         initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

            elif self.rnn_type == 'bilstm' or self.rnn_type == 'bigru':
                self.va = self.add_weight(name='va', shape=(4 * self.n_tensors, 1),
                                          initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

                self.vo =self.add_weight(name='vo', shape=(4 * self.n_tensors, 1),
                                         initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

            self.Ma = self.add_weight(name='Ma', shape=(self.n_hidden, self.n_hidden),
                                      initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

            self.Mo = self.add_weight(name='Mo', shape=(self.n_hidden, self.n_hidden),
                                      initializer=initializers.random_uniform(-0.2, 0.2), trainable=True)

        super(CMLA, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        """
        Feed forward phase.
        """
        def tensor_product(x):
            x = tf.expand_dims(x, 0)
            ma = self.m0_a
            mo = self.m0_o
            rnn_aspect = []
            rnn_opinion = []

            for i in range(self.n_layers):
                aspect_res1 = K.dot(x, K.dot(self.Ua, K.transpose(ma)))
                aspect_res2 = K.dot(x, K.dot(self.Va, K.transpose(mo)))

                opinion_res1 = K.dot(x, K.dot(self.Uo, K.transpose(ma)))
                opinion_res2 = K.dot(x, K.dot(self.Vo, K.transpose(mo)))

                aspect = K.tanh(K.squeeze(K.concatenate([aspect_res1, aspect_res2], axis=2), axis=3))
                opinion = K.tanh(K.squeeze(K.concatenate([opinion_res1, opinion_res2], axis=2), axis=3))

                rnn_aspect.append(self.rnn_aspect_layer(aspect))
                rnn_opinion.append(self.rnn_opinion_layer(opinion))

                if i == (self.n_layers - 1):
                  # return [K.sum(K.stack(gru_aspect), 0), K.sum(K.stack(gru_opinion), 0)]
                    return [K.squeeze(K.sum(K.stack(rnn_aspect), 0), axis=0), K.squeeze(K.sum(K.stack(rnn_opinion), 0), axis=0)]

                else:
                    alpha_aspect = K.softmax(K.squeeze(K.dot(rnn_aspect[i], self.va), axis=2))
                    ctx_pool_aspect = self.dot_layer([alpha_aspect, x])

                    alpha_opinion = K.softmax(K.squeeze(K.dot(rnn_opinion[i], self.vo), axis=2))
                    ctx_pool_opinion = self.dot_layer([alpha_opinion, x])

                    ma = K.tanh(K.dot(ma, self.Ma)) + ctx_pool_aspect
                    mo = K.tanh(K.dot(mo, self.Mo)) + ctx_pool_opinion

        return tf.map_fn(lambda x: tensor_product(x), x, dtype=[tf.float32, tf.float32])

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of this layer.
        """
        return [(input_shape[0], input_shape[1], 2 * self.n_tensors), (input_shape[0], input_shape[1], 2 * self.n_tensors)]

    def get_config(self):
        """
        Get layer config.
        """
        config = {'n_tensors': self.n_tensors,
                  'n_hidden': self.n_hidden,
                  'n_layers': self.n_layers,
                  'dropout_rate': self.dropout_rate,
                  'rnn_type': self.rnn_type}

        base_config = super(CMLA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))