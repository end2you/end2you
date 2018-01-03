from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def concordance_cc(predictions, labels):

    pred_mean, pred_var = tf.nn.moments(predictions, (0,))
    gt_mean, gt_var = tf.nn.moments(labels, (0,))

    mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (labels - gt_mean))

    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

def mse(pred_single, gt_single):
    return tf.reduce_mean(tf.square(pred_single - gt_single))

def get_loss(loss):
    return {
        'ccc': concordance_cc,
        'mse': mse,
    }[loss]
