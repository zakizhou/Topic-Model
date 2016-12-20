from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf


class Doc2Vec(object):
    def __init__(self, inputs):
        vocab_size = inputs.vocab_size
        input_size = inputs.input_size
        contexts = inputs.contexts
        sequence_length = inputs.sequence_length
        num_units = inputs.num_units

        with tf.variable_scope("embed"):
            embed = tf.get_variable(name="embed",
                                    shape=[vocab_size, input_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    dtype=tf.float32)
            lookup = tf.nn.embedding_lookup(embed, contexts)

        with tf.variable_scope("rnn"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units,
                                                state_is_tuple=True)
            _, state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=lookup,
                                         dtype=tf.float32,
                                         sequence_length=sequence_length)