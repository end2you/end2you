import tensorflow as tf

from src.losses import Losses
from src.models.model import Model
from src.data_provider.data_provider import DataProvider
from tensorflow.python.platform import tf_logging as logging
from pathlib import Path
from src.models.base import *

slim = tf.contrib.slim


class Train:
    
    def __init__(self,
                 predictions: tf.Tensor,
                 data_provider:DataProvider,
                 input_type:str,
                 train_dir: Path,
                 initial_learning_rate:float = 0.0001,
                 num_epochs:int = 100,
                 loss:str = 'ccc',
                 pretrained_model_checkpoint_path:Path = None):
        
        self.train_dir = str(train_dir)
        self.predictions = predictions
        self.input_type = input_type
        self.data_provider = data_provider
        self.initial_learning_rate = initial_learning_rate
        self.num_epochs = num_epochs
        self.loss = Losses.__dict__[loss.lower()]
        self.pretrained_model_checkpoint_path = \
                            str(pretrained_model_checkpoint_path)
    
    def _flatten(self, output, i):
        if self.data_provider.seq_length != None:
            return tf.reshape(output[:, :, i], (-1,))
        return tf.reshape(output[:, i], (-1,))
    
    def set_train_loss(self, prediction, label):
        
        train_loss = self.loss
        num_outputs = label.get_shape().as_list()[2] if self.data_provider.seq_length != None \
                else label.get_shape().as_list()[1]
        
        name_out = ['out'.format(i) for i in range(num_outputs)]
        
        for i, name in enumerate(name_out):
            pred_single = self._flatten(prediction, i)
            lb_single = self._flatten(label, i)
            
            loss = train_loss(lb_single, pred_single)
            tf.summary.scalar('losses/{}_loss'.format(name), loss)
            
            tf.losses.add_loss(loss / float(num_outputs))
    
    def restore_variables(self, scope=None):
        
        init_fn = None
        if self.pretrained_model_checkpoint_path:
            variables_to_restore = slim.get_model_variables(scope=scope)
            init_fn = slim.assign_from_checkpoint_fn(
                    self.pretrained_model_checkpoint_path, variables_to_restore)
        
        return init_fn
    
    def start_training(self):
        
        frames, labels, sids = self.data_provider.get_batch()
        
        predictions = self.predictions
        loss = self.set_train_loss(predictions, labels)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)
        
        optimizer = tf.train.AdamOptimizer(self.initial_learning_rate)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init_fn = self.restore_variables()
            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True)
            
            seq_length = 1 if self.data_provider.seq_length == None \
                           else self.data_provider.seq_length
            max_steps = self.num_epochs * self.data_provider.batch_size * seq_length
            logging.set_verbosity(1)
            slim.learning.train(train_op,
                                self.train_dir,
                                init_fn=init_fn,
                                save_summaries_secs=60,
                                save_interval_secs=300,
                                number_of_steps=max_steps)
