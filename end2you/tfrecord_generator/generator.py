import csv
import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")

from pathlib import Path
from io import BytesIO
from abc import ABCMeta, abstractmethod
from moviepy.editor import VideoFileClip, AudioFileClip
from ..rw.file_reader import FileReader
from functools import partial


class Generator(metaclass=ABCMeta):
    
    def __init__(self, 
                 reader: FileReader,
                 input_type:str,
                 delimiter:str = ';'):
        
        self.input_type = self._get_input_type(input_type.lower())
        
        self.attributes_name, self.attributes_type, self.data = \
                                                            reader.read()
        
        label_idx = self.attributes_name.index('label')
        file_idx = self.attributes_name.index('file')
        label_type = self.attributes_type[label_idx]
        
        kwargs = {}
        if label_type == 'str':
            read_label_file = FileReader.read_delimiter_file
            kwargs['delimiter'] = delimiter
        else:
            read_label_file = self._read_single_label
            kwargs['label_type'] = label_type
        
        self.dict_files = dict()

        for row in self.data[:, [file_idx, label_idx]]:
            data_file = row[0]
            label_file = row[1]
            if label_type != 'str':
                kwargs['file'] = data_file
                
            names, types, data = read_label_file(label_file, **kwargs)
            
            time_idx = names.index('time')
            labels_idx = np.delete(np.arange(0, len(data[0,:])), time_idx)
            
            num_labels = len(data[0, 1:])
            self.dict_files[data_file] = {
                                          'time': np.reshape(self._get_label_type(data[:, time_idx], types[0]), 
                                                             (-1,1)), 
                                          'labels': np.reshape(self._get_label_type(data[:, labels_idx], types[1]), 
                                                              (-1, num_labels))
                                         }
    
    def _get_label_type(self, label, _type):
        if 'float' in _type: 
            return list([np.float32(x) for x in label])
        return list([np.int32(x) for x in label])
    
    def _read_single_label(self, label, file=None, label_type=None):
        clip = VideoFileClip
        if 'audio' in self.input_type:
            clip = AudioFileClip
        end_time = clip(str(file)).duration
        
        time = np.vstack([0.0, end_time])
        label = np.reshape(np.repeat(self._get_label_type(label, label_type), 2), (-1, 1))

        return ['time', 'labels'], ['float', label_type], np.reshape(np.hstack( [time, label]) , (-1, 2))
    
    def _get_input_type(self, input_type):
        correct_types = ['audio','video','audiovisual']
        if input_type not in correct_types:
            raise ValueError('input_type should be one of {}'.format(correct_types),
                             '[{}] found'.format(input_type))
        return input_type        
    
    def _int_feauture(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feauture(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def write_tfrecords(self, tfrecords_folder):
        
        print('\n Start generating tf records \n')
        
        for data_file in self.dict_files.keys():
            print('Writing file : {}'.format(data_file))
                
            basename = os.path.basename(os.path.splitext(data_file)[0])
            writer = tf.python_io.TFRecordWriter(
                (Path(tfrecords_folder) / '{}.tfrecords'.format(basename)
                ).as_posix())
            
            self.serialize_sample(writer, data_file, basename)
    
    @abstractmethod
    def _get_samples(self, data_file):
        pass
    
    @abstractmethod
    def serialize_sample(self, writer, data_file, subject_id):
        pass