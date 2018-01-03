import tensorflow as tf

slim = tf.contrib.slim


def fully_connected(inputs:tf.Tensor,
                   number_of_outputs:int):
    return slim.layers.linear(inputs, number_of_outputs)


def combined_model(*args):
    return tf.concat((*args), 2, name='concat')

