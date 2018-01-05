import tensorflow as tf
import numpy as np

from abc import ABCMeta, abstractmethod
from pathlib import Path


class DataProvider(metaclass=ABCMeta):
    
    def __init__(self,
                 tfrecords_folder: Path,
                 is_training:bool = True, 
                 split_name:str = 'train', 
                 batch_size:int = 32,
                 seq_length:int = None):
        
        tf_records = [str(x) for x in tfrecords_folder.glob('*.tfrecords')][0]
        
        self.frame_shape = self.get_shape(tf_records, 'frame')
        self.label_shape = self.get_shape(tf_records, 'label')
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.is_training = is_training
        
        self.root_path = Path(tfrecords_folder)
        paths = [str(x) for x in self.root_path.glob('*.tfrecords')]
        
        filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)

        reader = tf.TFRecordReader()

        _, self.serialized_example = reader.read(filename_queue)
        
    @abstractmethod
    def parse_and_decode_example(self):
        pass
    
    def get_shape(self, input_files, tensor_name):
        record_iterator = tf.python_io.tf_record_iterator(path=str(input_files))
        
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            tt = (example.features.feature[tensor_name]
                                         .bytes_list
                                         .value[0])
            input_shape = np.fromstring(tt, dtype=np.float32)
            break
        
        return input_shape.shape[0]

    def augment_data(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_batch(self):
        pass
