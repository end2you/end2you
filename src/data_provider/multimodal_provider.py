import tensorflow as tf

from pathlib import Path
from inception_processing import distort_color


class MultimodalProvider(DataProvider):
    
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def parse_and_decode_example(self):
        features = tf.parse_single_example(
            self.serialized_example,
            features={
                'sample_id': tf.FixedLenFeature([], tf.int64),
                'subject_id': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.float32),
                'raw_audio': tf.FixedLenFeature([], tf.float32),
                'frame': tf.FixedLenFeature([], tf.float32),
            }
        )

        frame = tf.decode_raw(features['frame'], tf.float32)
        raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
        label = tf.decode_raw(features['label'], self.get_tf_type)
        subject_id = features['subject_id']
        
        return frame, raw_audio, label, subject_id
    
    def get_batch(self):
        frame, raw_audio, label, subject_id = self.parse_and_decode_example()
        
        frame.set_shape(self.frame_shape)
        label.set_shape(self.label)
        raw_audio.set_shape(self.raw_audio)
        
        if self.seq_length == None:
            return self._get_single_example_batch(frame, raw_audio, label, subject_id)
        else:
            return self._get_seq_examples_batch(frame, raw_audio, label, subject_id)
        
    def _get_single_example_batch(self, frame, raw_audio, label, subject_id):

        frames, raw_audio, labels, subject_ids = \
            tf.train.batch([frame, raw_audio, label, subject_id], self.batch_size,
                            capacity=1000, dynamic_pad=True)
        
        return frames, raw_audio, labels, subject_ids
    
    def _get_seq_examples_batch(self, frame, label, subject_id):
        # Number of threads should always be one, in order to load samples
        # sequentially.
        frames, raw_audio, labels, subject_ids = tf.train.batch(
            [frame, raw_audio, label, subject_id], self.seq_length, num_threads=1, capacity=1000)
    
        labels = tf.expand_dims(labels, 0)
        frames = tf.expand_dims(frames, 0)
        raw_audio = tf.expand_dims(raw_audio, 0)
        
        if self.is_training:
            frames, raw_audio, labels, subject_ids = tf.train.shuffle_batch(
                [frames, raw_audio, labels, subject_ids], batch_size, 1000, 50, num_threads=1)
        else:
            frames, raw_audio, labels, subject_ids = tf.train.batch(
                [frames, raw_audio, labels, subject_ids], batch_size, num_threads=1, capacity=1000)
            
        return frames[:, 0, :, :], raw_audio[:, 0, :, :], labels[:, 0, :, :], subject_ids
