import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append("..")

from moviepy.audio.io.AudioFileClip import AudioFileClip
from ..models.model import Model
from pathlib import Path
from ..rw.file_reader import FileReader


slim = tf.contrib.slim

class TestRaw:
    
    def __init__(self, 
                 reader:FileReader,
                 model_path:Path,
                 predictions:Model,
                 input_type:str = 'audio',
                 task:str = 'classification',
                 prediction_file:Path = Path('predictions.csv'),
                 frame_shape:list = [1, None, 640]):
        
        self.model_path = str(model_path)
        self.attributes_name, self.attributes_type, self.data_files = reader.read()
        self.task = task
        self.prediction_file = str(prediction_file)
        file_idx = self.attributes_name.index('file')
        self.data_files = np.array(self.data_files[:, [file_idx]])
        
        if input_type == 'audio':
            self.reader = self.read_single_wav
        else:
            raise ValueError('Only audio testing is supported.')
        
        self.frames = tf.placeholder(tf.float32, frame_shape)
        self.predictions = predictions
        
    @staticmethod
    def get_data(data_files, reader):
        """ Returns the raw input and labels. 
        
        Args:
          dataset_dir: The directory that contains the data.
          reader: Reads the data.
        Returns:
          The raw examples and the corresponding labels.
        """
        for i, name in enumerate(data_files):
            data = reader(name[0])
            
            if isinstance(data, list):
                for d in data:
                    yield d, name
            else:
                yield data, name
    
    def read_single_wav(self, path):
        clip = AudioFileClip(str(path))

        subsampled_audio = clip.set_fps(16000)
        chunk_size = 640

        audio = np.array(list(subsampled_audio.iter_frames())).mean(1)
        audio = np.pad(audio, (0, chunk_size - audio.shape[0] % chunk_size), 'constant')
        audio = audio.reshape(-1, chunk_size)
        
        return audio.astype(np.float32)
    
    def start_testing(self):
        
        predictions = self.predictions(self.frames)

        coord = tf.train.Coordinator()
        variables_to_restore = slim.get_variables_to_restore()
        
        evaluated_predictions = []
        test_files_name = []

        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            print('\nLoading model [{}]\n'.format(self.model_path))
            saver.restore(sess, self.model_path)
            tf.train.start_queue_runners(sess=sess)
            
            print('Start getting predictions.\n')
            for data, name in self.get_data(self.data_files, self.reader):
                print('Getting prediction for file : {}'.format(name))
                
                data = np.expand_dims(data, 0)
                pr = sess.run(predictions, feed_dict={self.frames:data})
                
                evaluated_predictions.append(pr)
                test_files_name.append(name)
                
                if coord.should_stop():
                    break
            
            print('\nEnd getting predictions.')
            coord.request_stop()

            predictions = np.reshape(evaluated_predictions, (-1, pr.shape[-1]))
            test_files = np.reshape(np.array(test_files_name), (-1, 1))
            
            if self.task == 'classification':
                predictions = np.reshape(np.argmax(predictions, axis=1), (-1, 1))
            
            output = np.hstack((test_files, predictions))
            
            print('\nWriting predictions to file : {}'.format(self.prediction_file))
            np.savetxt(str(self.prediction_file), output, delimiter=",", fmt='%s' )

            return output