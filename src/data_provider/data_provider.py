import tensorflow as tf

from abc import ABCMeta, abstractmethod
from pathlib import Path


class DataProvider(metaclass=ABCMeta):
    
    def __init__(self,
                 tfrecords_folder: Path, 
                 frame_shape:list,
                 label_shape:list,
                 is_training:bool = True, 
                 split_name:str = 'train', 
                 batch_size:int = 32,
                 seq_length:int = None):
        
        self.frame_shape = frame_shape
        self.label_shape = label_shape
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.is_training = is_training
        
        self.root_path = Path(tfrecords_folder) / split_name
        paths = [str(x) for x in self.root_path.glob('*.tfrecords')]
        
        filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)

        reader = tf.TFRecordReader()

        _, self.serialized_example = reader.read(filename_queue)
        
    @abstractmethod
    def parse_and_decode_example(self):
        pass
    
    def augment_data(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_batch(self):
        pass
