import tensorflow as tf
import sys
sys.path.append("..")

from .metrics import concordance_cc2 as m_concordance_cc2
from ..models.model import Model
from ..data_provider.data_provider import DataProvider
from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

class Eval:
    
    def __init__(self, 
                 train_dir:Path,
                 log_dir:Path,
                 predictions:Model,
                 data_provider:DataProvider,
                 num_examples:int,
                 seq_length:int = None,
                 metric:str = 'ccc',
                 eval_interval_secs:int = 300
                ):
        
        
        self.eval_interval_secs = eval_interval_secs
        self.predictions = predictions
        self.data_provider = data_provider
        self.train_dir = str(train_dir)
        self.log_dir = str(log_dir)
        self.predictions = predictions
        self.metric = [metric] #[x for x in kwargs['metric'].split(',')]
        self.seq_length = seq_length
        
        self.name_pred = ['pred_{}'.format(i) for i in range(self.data_provider.label_shape[0])]
        
        self.num_examples = num_examples
        self.num_outputs = data_provider.label_shape
        
    def _create_summary(self, name, value):
        op = tf.summary.scalar(name, value)
        op = tf.Print(op, [value], name)
        return op
    
    def get_mse(self, predictions, labels):
        
        metric = {}
        for i, name in enumerate(self.name_pred):
            key = 'eval/{}'.format(name)
            metric[key] = slim.metrics.streaming_mean_squared_error(predictions[:,:,i], 
                                                                    labels[:,:,i])
            
        summary_ops = []
        mse_total = 0
        
        # Computing MSE and Concordance values, and adding them to summary
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric)
        for i, name in enumerate(self.name_pred):
            key = 'eval/{}'.format(name)
            op = self._create_summary(key, names_to_values[key])
            summary_ops.append(op)
        
            mse_total += names_to_values[key]
        
        mse_total = mse_total / self.num_outputs
        
        op = self._create_summary('eval/mse_total', mse_total)
        summary_ops.append(op)
        
        return names_to_values, names_to_updates, summary_ops
    
    def get_ccc(self, predictions, labels):
        
        summary_ops = []
        names_to_updates = {}
        conc_total = 0
        for i, name in enumerate(self.name_pred):
            with tf.name_scope(name) as scope:
                concordance_cc2, values, updates = m_concordance_cc2(
                            tf.reshape(predictions[:,:,i], [-1]),
                            tf.reshape(labels[:,:,i], [-1]))
            
            for n, v in updates.items():
                names_to_updates[n + '/' + name] = v
          
            op = self._create_summary('eval/concordance_' + name, concordance_cc2)
            summary_ops.append(op)
          
            conc_total += concordance_cc2

        conc_total = conc_total / self.num_outputs
        
        op = self._create_summary('eval/concordance_total', conc_total)
        summary_ops.append(op)
        
        return {}, names_to_updates, summary_ops
    
    def get_metric(self, metric):
        return {
            'ccc':self.get_ccc,
            'mse':self.get_mse
        }[metric]
    
    def start_evaluation(self):
        
        frames, labels, sids = self.data_provider.get_batch()
        
        predictions = self.predictions

        names_to_values = {}
        names_to_updates = {}
        summary_ops = []
        for m in self.metric:
            values, updates, s_ops = self.get_metric(m)(predictions, labels)
            names_to_values.update(values)
            names_to_updates.update(updates)
            summary_ops.extend(s_ops)
        
        seq_length = 1 if self.data_provider.seq_length == None \
                       else self.data_provider.seq_length
            
        num_batches = int(self.num_examples / (1 * seq_length))
        logging.set_verbosity(1)

        # Setup the global step.
        eval_interval_secs = self.eval_interval_secs # How often to run the evaluation.
        slim.evaluation.evaluation_loop(
            '',
            self.train_dir,
            self.log_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs)
