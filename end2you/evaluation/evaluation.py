import tensorflow as tf
import sys
sys.path.append("..")

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
        self.seq_length = 1 if data_provider.seq_length == 0 \
            else data_provider.seq_length
        
        self.num_outputs = data_provider.label_shape
        
        if 'uar' in self.metric:
            self.num_outputs = self.data_provider.num_classes
            
        self.name_pred = ['pred_{}'.format(i) for i in range(self.data_provider.label_shape[0])]
        self.num_examples = self.data_provider.num_examples
        
        
    def _create_summary(self, name, value):
        op = tf.summary.scalar(name, value)
        op = tf.Print(op, [value], name)
        return op 
    