import tensorflow as tf

from pathlib import Path
from .data_provider import DataProvider


@DataProvider.register
class MultimodalProvider(DataProvider):
    
    def __init__(self, *args, **kwargs):
        
        p = Path([*args][0])
        if 'tf_records_folder' in list(kwargs.keys()):
            f = kwargs['tf_records_folder']
        f = [str(x) for x in p.glob('*.tfrecords')][0]
        self.raw_audio_shape = list(self.get_shape(f, 'raw_audio_shape'))
        
        super().__init__(*args, **kwargs)
    
    def parse_and_decode_example(self):
        features = tf.parse_single_example(
            self.serialized_example,
            features={
                'sample_id': tf.FixedLenFeature([], tf.int64),
                'subject_id': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'raw_audio': tf.FixedLenFeature([], tf.string),
                'frame': tf.FixedLenFeature([], tf.string)
            }
        )
        
        frame = tf.decode_raw(features['frame'], tf.float32) / 255.
        raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
        label = tf.decode_raw(features['label'], self._get_tf_type())
        subject_id = features['subject_id']

        frame = tf.reshape(frame, self.frame_shape)
        raw_audio = tf.reshape(raw_audio, self.raw_audio_shape)
        
        if self.is_training and self.noise:
            raw_audio += tf.random_normal(tf.shape(raw_audio), stddev=self.noise)
        
        return frame, raw_audio, label, subject_id
    
    def get_batch(self):
        frame, raw_audio, label, subject_id = self.parse_and_decode_example()
    
        label.set_shape(self.label_shape)
        raw_audio.set_shape(list(self.raw_audio_shape))
        frame.set_shape(self.frame_shape)
        
        if self.seq_length == None:
            return self._get_single_example_batch(frame, raw_audio, label, subject_id)
        else:
            return self._get_seq_examples_batch(frame, raw_audio, label, subject_id)