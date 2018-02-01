import tensorflow as tf
import numpy as np

from . import metrics
from .evaluation import Eval

class EvalOnce(Eval):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def get_metric(metric):
        return {
            'ccc':metrics.np_concordance_cc2,
            'mse':metrics.np_mse,
            'uar':metrics.np_uar
        }[metric]
    
    @staticmethod
    def get_eval_tensors(sess, predictions, data_provider, evalute_path):
        
        data_provider.tfrecords_folder = evalute_path 
        num_examples = data_provider.get_num_examples(evalute_path)
        frames, labels, sids = data_provider.get_batch()
        
        get_pred = predictions(frames)
        
        seq_length = 1 if data_provider.seq_length == 0 \
            else data_provider.seq_length
        
        num_batches = int(np.ceil(
            data_provider.num_examples / (data_provider.batch_size * seq_length)))

        return get_pred, labels, num_batches
    
    @staticmethod
    def eval_once(sess, get_pred, labels, num_batches, num_outputs, metric_name):
        
        metric = EvalOnce.get_metric(metric_name)
        
        print('\n Start Evaluation \n')
        evaluated_predictions = []
        evaluated_labels = []
        for batch in range(num_batches):
            print('Example {}/{}'.format(batch+1, num_batches))
            preds, labs = sess.run([get_pred, labels])
            evaluated_predictions.append(preds)
            evaluated_labels.append(labs)
        
        predictions = np.reshape(evaluated_predictions, (-1, num_outputs))
        labels = np.reshape(evaluated_labels, (-1, num_outputs))
        
        mean_eval = metric(predictions, labels)
        
        return mean_eval
    
    
