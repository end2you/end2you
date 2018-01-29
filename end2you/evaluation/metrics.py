import tensorflow as tf
import numpy as np

from sklearn.metrics import recall_score


slim = tf.contrib.slim

def tf_concordance_cc2(prediction, ground_truth):

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'eval/mean_pred':slim.metrics.streaming_mean(prediction),
      'eval/mean_lab':slim.metrics.streaming_mean(ground_truth),
      'eval/cov_pred':slim.metrics.streaming_covariance(prediction, prediction),
      'eval/cov_lab':slim.metrics.streaming_covariance(ground_truth, ground_truth),
      'eval/cov_lab_pred':slim.metrics.streaming_covariance(prediction, ground_truth)
    })

    metrics = dict()
    
    for name, value in names_to_values.items():
        metrics[name] = value

    mean_pred = metrics['eval/mean_pred']
    var_pred = metrics['eval/cov_pred']
    mean_lab = metrics['eval/mean_lab']
    var_lab = metrics['eval/cov_lab']
    var_lab_pred = metrics['eval/cov_lab_pred']

    denominator = (var_pred + var_lab + (mean_pred - mean_lab) ** 2)

    concordance_cc2 = (2 * var_lab_pred) / denominator

    return concordance_cc2, names_to_values, names_to_updates

def np_concordance_cc2(r1, r2):
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)

def np_uar(predictions, labels):
    labels = np.argmax(labels, axis = 1)
    predictions = np.argmax(predictions, axis = 1)
        
    return recall_score(labels, predictions, average="macro")

def np_mse(predictions, labels):
    return np.mean( np.sqrt(predictions - labels))