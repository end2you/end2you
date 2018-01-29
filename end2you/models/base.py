import tensorflow as tf

slim = tf.contrib.slim


def fully_connected(inputs:tf.Tensor,
                   number_of_outputs:int):
    with tf.variable_scope("fully_connected", reuse=tf.AUTO_REUSE):
        fc = slim.layers.linear(inputs, number_of_outputs)
    return fc


def combine_models(*args):
    return tf.concat((*args), 2, name='concat')

