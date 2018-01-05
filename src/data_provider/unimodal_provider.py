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
        
        frame = tf.decode_raw(features['frame'], tf.float32)
        label = tf.decode_raw(features['label'], self._get_tf_type())
        subject_id = features['subject_id']
    
        return frame, label, subject_id
    
    def get_batch(self):
        frame, label, subject_id = self.parse_and_decode_example()
        
        frame.set_shape(self.frame_shape)
        label.set_shape(self.label_shape)
        
        if self.seq_length == None:
            return self._get_single_example_batch(frame, label, subject_id)
        else:
            return self._get_seq_examples_batch(frame, label, subject_id)
        
    def _get_single_example_batch(self, frame, label, subject_id):

        frames, labels, subject_ids = \
            tf.train.batch([frame, label, subject_id], self.batch_size,
                            capacity=1000, dynamic_pad=True)
        
        return frames, labels, subject_ids
    
    def _get_seq_examples_batch(self, frame, label, subject_id):
        # Number of threads should always be one, in order to load samples
        # sequentially.
        frames, labels, subject_ids = tf.train.batch(
            [frame, label, subject_id], self.seq_length, num_threads=1, capacity=1000)
    
        labels = tf.expand_dims(labels, 0)
        frames = tf.expand_dims(frames, 0)
        
        if self.is_training:
            frames, labels, subject_ids = tf.train.shuffle_batch(
                [frames, labels, subject_ids], self.batch_size, 1000, 50, num_threads=1)
        else:
            frames, labels, subject_ids = tf.train.batch(
                [frames, labels, subject_ids], self.batch_size, num_threads=1, capacity=1000)
            
        return frames[:, 0, :, :], labels[:, 0, :, :], subject_ids
