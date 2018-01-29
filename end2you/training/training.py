import tensorflow as tf
import sys
sys.path.append("..")

from .losses import Losses
from ..models.model import Model
from ..data_provider.data_provider import DataProvider
from ..models.base import *
from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim


class Train:
    
    def __init__(self,
                 predictions: tf.Tensor,
                 data_provider:DataProvider,
                 train_dir: Path = 'ckpt/train',
                 log_dir: Path = 'ckpt/log',
                 initial_learning_rate:float = 0.0001,
                 num_epochs:int = 100,
                 loss:str = 'ccc',
                 pretrained_model_checkpoint_path:Path = None,
                 save_top_k:int = 5):
        
        self.log_dir = log_dir
        self.train_dir = train_dir
        self.predictions = predictions
        self.data_provider = data_provider
        self.initial_learning_rate = initial_learning_rate
        self.num_epochs = num_epochs
        self.loss = Losses.__dict__[loss.lower()]
        self.pretrained_model_checkpoint_path = \
                            str(pretrained_model_checkpoint_path)
    
    def _flatten(self, output, i):
        if self.data_provider.seq_length != 0:
            return tf.reshape(output[:, :, i], (-1,))
        return tf.reshape(output[:, i], (-1,))
    
    def set_train_loss(self, prediction, label):
        
        train_loss = self.loss
        self.num_outputs = prediction.get_shape().as_list()[1] if self.data_provider.seq_length == 0 \
                    else prediction.get_shape().as_list()[2]
        
        if self.loss.__name__ == 'mse' or self.loss.__name__ == 'concordance_cc':
            
            name_out = ['out_{}'.format(i) for i in range(self.num_outputs)]
            
            for i, name in enumerate(name_out):
                pred_single = self._flatten(prediction, i)
                lb_single = self._flatten(label, i)
                
                loss = train_loss(lb_single, pred_single)
                tf.summary.scalar('losses/{}_loss'.format(name), loss)

                tf.losses.add_loss(loss / float(num_outputs))
        else:
            loss = train_loss(label, prediction)
            tf.summary.scalar('losses/CE_loss', loss)