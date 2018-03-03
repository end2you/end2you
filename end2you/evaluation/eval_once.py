import tensorflow as tf
import numpy as np
import copy

from . import metrics
from .evaluation import Eval
from pathlib import Path

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

        dp_eval = copy.copy(data_provider)
        paths = [str(x) for x in Path(evalute_path).glob('*.tfrecords')]
        filename_queue = tf.train.string_input_producer(paths, shuffle=False)
#        dp_eval.tfrecords_folder = evalute_path
        dp_eval.num_examples = dp_eval.get_num_examples(evalute_path)

        _, dp_eval.serialized_example = tf.TFRecordReader().read(filename_queue)

        frames, labels, sids = dp_eval.get_batch()

        get_pred = predictions(frames)
        
        seq_length = 1 if data_provider.seq_length == 0 \
            else data_provider.seq_length
        
        num_batches = int(np.ceil(
            dp_eval.num_examples / (dp_eval.batch_size * seq_length)))

        return get_pred, labels, sids, num_batches
    
    @staticmethod
    def eval_once(sess, get_pred, labels, sids, num_batches, num_outputs, metric_name):
        
        metric = EvalOnce.get_metric(metric_name)
        
        print('\n Start Evaluation \n')
        evaluated_predictions = []
        evaluated_labels = []
        for batch in range(num_batches):
            print('Example {}/{}'.format(batch+1, num_batches))
            preds, labs, s = sess.run([get_pred, labels, sids])
            evaluated_predictions.append(preds)
            evaluated_labels.append(labs)
        
        predictions = np.reshape(evaluated_predictions, (-1, num_outputs))
        labels = np.reshape(evaluated_labels, (-1, num_outputs))
        
        mean_eval = metric(predictions, labels)
        
        return mean_eval
    
    
