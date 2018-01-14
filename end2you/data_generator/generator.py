import csv
import os
import numpy as np
import tensorflow as tf

from pathlib import Path
from io import BytesIO
from abc import ABCMeta, abstractmethod
from moviepy.editor import VideoFileClip, AudioFileClip

class Generator(metaclass=ABCMeta):
    
    def __init__(self, 
                 data_file:Path,
                 input_type:str,
                 task:str = 'regression',
                 delimiter:str = ';'):
        
        self.task = task.lower()
        self.data = np.loadtxt(data_file, delimiter=delimiter, dtype=str)
        self.input_type = self._get_input_type(input_type.lower())
        
        read_label_file = self._read_csv_file
        
        if len(self.data[0][1]) == 1:
            read_label_file = self._read_single_label
        
        self.dict_files = dict()
        for data_file, label_file in self.data:
            time, labels = read_label_file(label_file, data_file, delimiter=delimiter)
            self.dict_files[data_file] = {
                                          'time': time, 
                                          'labels': labels
                                         }
    
    def add_arguments(self, parser):
        parser.add_argument('--data_file', type=str,
                    help='Contains the file paths to convert to tf records'
                         'along with their labels, separated with `;`.')
        
        return parser
    
    def _get_label_type(self, label):
        if 'regression' in self.task:
            return list([np.float32(x) for x in label])
        return list([np.int32(x) for x in label])
    
    def _read_single_label(self, label, data_file, delimiter=None):
        clip = VideoFileClip
        if 'audio' in self.input_type:
            clip = AudioFileClip
        end_time = clip(str(data_file)).duration
        
        return np.array([0.0, end_time]), np.array([self._get_label_type(label)])
    
    def _get_input_type(self, input_type):
        correct_types = ['audio','video','audiovisual']
        if input_type not in correct_types:
            raise ValueError('input_type should be one of {}.'format(correct_types),
                             '[{}] found'.format(input_type))
        return input_type        
    
    def _int_feauture(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feauture(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _read_csv_file(self, labels_file, data_file=None,delimiter=';', skiprows=[0]):
        with open(labels_file, 'r') as lf:
            csvreader = csv.reader(lf, delimiter=delimiter)
            time = []
            labels = []
            for i, row in enumerate(csvreader):
                if i in skiprows:
                    continue
                
                time.append(float(row[0]))
                labels.append(self._get_label_type(row[1:]))
        
        time = np.array(time)
        labels = np.array(labels)
        
        return time, labels
    
    def write_tfrecords(self, tfrecords_folder):
        for data_file in self.dict_files.keys():
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