import tensorflow as tf

from enum import Enum


slim = tf.contrib.slim

class Losses(Enum):
    
    def concordance_cc(ground_truth, predictions):

        pred_mean, pred_var = tf.nn.moments(predictions, (0,))
        gt_mean, gt_var = tf.nn.moments(ground_truth, (0,))

        mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (ground_truth - gt_mean))

        return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))
    
    def mse(ground_truth, predictions):
        return tf.reduce_mean(tf.square(predictions - ground_truth))
    
    ccc = concordance_cc
    mse = mse
    ce = tf.losses.softmax_cross_entropy