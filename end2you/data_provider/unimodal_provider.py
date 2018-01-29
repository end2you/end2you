import tensorflow as tf

from pathlib import Path
from .data_provider import DataProvider

slim = tf.contrib.slim

@DataProvider.register
class UnimodalProvider(DataProvider):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def parse_and_decode_example(self):
        features = tf.parse_single_example(
            self.serialized_example,
            features={
                'sample_id': tf.FixedLenFeature([], tf.int64),
                'subject_id': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'frame': tf.FixedLenFeature([], tf.string),
            }
        )
        
        frame = tf.decode_raw(features['frame'], tf.float32) 
        label = tf.decode_raw(features['label'], self._get_tf_type())
        subject_id = features['subject_id']
        
        if self.seq_length != 0:
            frame = tf.reshape(frame, self.frame_shape)
        
        return frame, label, subject_id
    
    def get_batch(self):
        frame, label, subject_id = self.parse_and_decode_example()
        
        if self.seq_length == 0: # assume for audio input only
            frames, labels, subjects_id = \
                self._get_single_example_batch(self.batch_size, frame, label, subject_id)
            
            if 'classification' in self.task:
                labels = tf.squeeze(labels, axis=1)
                labels = slim.one_hot_encoding(labels, self.num_classes)
            
            frames = tf.reshape(frames, (self.batch_size, -1, 640)) 
            
            return frames, labels, subjects_id
        else:
            frame.set_shape(self.frame_shape)
            label.set_shape(self.label_shape)
            
            if 'classification' in self.task:
                labels = slim.one_hot_encoding(labels, self.num_classes)
                
            return self._get_seq_examples_batch(frame, label, subject_id)