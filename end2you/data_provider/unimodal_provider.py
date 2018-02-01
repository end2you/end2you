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
        
        frame = features['frame']
        label = features['label']
        subject_id = features['subject_id']
        
        if self.is_training:
            frame, label, subject_id = tf.train.shuffle_batch(
                    [frame, label, subject_id], 1, 1000, 100, 4)
            frame = frame[0]
            label = label[0]
            
        frame = tf.decode_raw(frame, tf.float32) 
        label = tf.decode_raw(label, self._get_tf_type())

        if self.seq_length != 0:
            frame = tf.reshape(frame, self.frame_shape)
        
        if self.is_training and self.noise:
            frame += tf.random_normal(tf.shape(frame), stddev=self.noise)
            
        return frame, label, subject_id
    
    def get_batch(self):
        frame, label, subject_id = self.parse_and_decode_example()
        
        if self.seq_length == 0: # assume for audio input only
            frames, labels, subjects_id = \
                self._get_single_example_batch(self.batch_size, frame, label, subject_id)
            
            if 'classification' in self.task:
                labels = slim.one_hot_encoding(labels, self.num_classes)
                labels = tf.reshape(labels, (self.batch_size, self.num_classes))

            frames = tf.reshape(frames, (self.batch_size, -1, 640)) 
            
            return frames, labels, subjects_id
        else:
            frame.set_shape(self.frame_shape)
            label.set_shape(self.label_shape)
            
            if 'classification' in self.task:
                labels = slim.one_hot_encoding(labels, self.num_classes)
                
            return self._get_seq_examples_batch(frame, label, subject_id)
