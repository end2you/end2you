import tensorflow as tf
import numpy as np

from .losses import Losses
from ..data_provider.data_provider import DataProvider
from tensorflow.python.platform import tf_logging as logging
from .training import Train

slim = tf.contrib.slim


class SlimTraining(Train):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def restore_variables(self, scope=None):
        
        init_fn = None
        if self.pretrained_model_checkpoint_path != 'None':
            variables_to_restore = slim.get_model_variables(scope=scope)
            init_fn = slim.assign_from_checkpoint_fn(
                    self.pretrained_model_checkpoint_path, variables_to_restore)
        
        return init_fn
    
    def start_training(self):
        
        frames, labels, sids = self.data_provider.get_batch()
        
        if self.data_provider.seq_length != 0:
            frames_shape = frames.get_shape().as_list()
            batch = self.data_provider.seq_length * self.data_provider.batch_size
            frames = tf.reshape(frames, [batch, *frames_shape[2:]] )
        
        predictions = self.predictions(frames)
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
            
            seq_length = 1 if self.data_provider.seq_length == 0 \
                           else self.data_provider.seq_length
            max_steps = np.ceil(self.num_epochs * \
                    self.data_provider.num_examples / (self.data_provider.batch_size * seq_length))
            
            logging.set_verbosity(1)
            slim.learning.train(train_op,
                                self.train_dir,
                                init_fn=init_fn,
                                save_summaries_secs=60,
                                save_interval_secs=300,
                                number_of_steps=max_steps)
