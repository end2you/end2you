import tensorflow as tf

from pathlib import Path
from .data_provider import DataProvider


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
        
        frame = tf.decode_raw(features['frame'], tf.float32) / 255.
        label = tf.decode_raw(features['label'], self._get_tf_type())
        subject_id = features['subject_id']
        
        frame = tf.reshape(frame, self.frame_shape)
        
        return frame, label, subject_id
    
    def get_batch(self):
        frame, label, subject_id = self.parse_and_decode_example()
        
        frame.set_shape(self.frame_shape)
        label.set_shape(self.label_shape)
        
        if self.seq_length == None:
            return self._get_single_example_batch(1, frame, label, subject_id)
        else:
            return self._get_seq_examples_batch(frame, label, subject_id)
    