import csv
import os
import numpy as np
import tensorflow as tf

from pathlib import Path
from io import BytesIO
from abc import ABCMeta, abstractmethod

class Generator(metaclass=ABCMeta):
    
    def __init__(self, 
                 data_file:Path,
                 input_type:str,
                 #task:str,
                 delimiter:str = ';'):
        
        self.data = np.loadtxt(data_file, delimiter=delimiter, dtype=str)
        self.dict_files = dict()
        
        for data_file, label_file in self.data:
            self.time, self.labels = self._read_csv_file(label_file, delimiter=delimiter)
            self.dict_files[data_file] = {
                                          'time': self.time, 
                                          'labels': self.labels
                                         }
        self.input_type = input_type
        #self.task = task
        
    def _int_feauture(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feauture(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _read_csv_file(self, labels_file, delimiter=';', skiprows=[0]):
        with open(labels_file, 'r') as lf:
            csvreader = csv.reader(lf, delimiter=delimiter)
            time = []
            labels = []
            for i, row in enumerate(csvreader):
                if i in skiprows:
                    continue
                
                time.append(float(row[0]))
                labels.append([float(x) for x in row[1:]])
        
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