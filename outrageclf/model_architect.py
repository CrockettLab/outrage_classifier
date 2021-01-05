"""

Specification of Deep and Transfer Learning model, as called in training.py.

Model Architect includes:
- Deep LSTM
- Deep GRU
- Bidirectional LSTM
- Bidirectional with Attention

We only provide access to our Deep GRU model, as used in the paper:

"""

from sklearn import ensemble
import tensorflow as tf
import keras
import keras.layers as layers
import tensorflow.keras.backend as K
from keras import Sequential, optimizers, initializers, regularizers, constraints
from keras.engine.topology import Layer


embedding_dim = 50
maxlen = 50


'''
Threshold function
'''

def threshold_acc(y_true, y_pred, threshold = 0.7):
    if K.backend() == 'tensorflow':
        return K.mean(K.equal(y_true,
          K.cast(K.greater_equal(y_pred,threshold), y_true.dtype)))
    else:
        return K.mean(K.equal(y_true,
          K.greater_equal(y_pred,threshold)))
    
 

'''
3-Layer LSTM model: 128, 64, 1 units each
2 Dropout layers
'''

def lstm_model (embedding_matrix, vocab_size):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=True))
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[threshold_acc])
    return (model)



'''
Deep GRU model: 256, 128, 64, 32 layer
2 Dropout layers
'''

def deep_gru_model (embedding_matrix, vocab_size):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=True))
    model.add(layers.GRU(256, return_sequences = True))
    model.add(layers.GRU(128, return_sequences = True))
    model.add(layers.GRU(64, return_sequences = True))
    model.add(layers.GRU(32))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[threshold_acc])
    return (model)



'''
Bi-directional model
'''

def deep_bidirectional_model (embedding_matrix, vocab_size):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix],
                               input_length=maxlen, trainable=True))
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences = True)))
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences = True)))
    model.add(layers.Bidirectional(layers.GRU(64)))

    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[threshold_acc])
    return (model)


'''
Architecture for Attention layer
including: - dot_product wrapper
           - predefined AttentionWithContext layer
'''

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]



'''
Attention model is a Bidirectional GRU with a layer of Attention
'''

def attention_model (embedding_matrix, vocab_size):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=True))
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences = True)))
    model.add(layers.Bidirectional(layers.GRU(64, return_sequences = True)))
    model.add(AttentionWithContext())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[threshold_acc])
    return model
