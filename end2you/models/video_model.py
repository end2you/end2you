import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1
from .model import Model


slim = tf.contrib.slim

class VideoModel(Model):
    
    def __init__(self,
                is_training:bool = True):
        
        self.is_training = is_training
    
    def create_model(self, frames):
        with tf.variable_scope("video_model", reuse=tf.AUTO_REUSE):
            with slim.arg_scope(slim.nets.resnet_utils.resnet_arg_scope()):
                video_input = tf.cast(frames, tf.float32)

                features, _ = resnet_v1.resnet_v1_50(video_input, None, self.is_training)

        return features