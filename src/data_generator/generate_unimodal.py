import os
import tensorflow as tf
import numpy as np

from src.data_generator.generator import Generator
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path


class UnimodalGenerator(Generator):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         **kwargs)
    
    def _get_samples(self, data_file):
        
        time = self.dict_files[data_file]['time']
        
        if 'audio' in self.input_type.lower():
            audio_clip = AudioFileClip(str(data_file))
            clip = audio_clip.set_fps(16000)
            num_samples = int(clip.fps * (time[1] - time[0]))
        elif 'video' in self.input_type.lower():
            clip = VideoFileClip(str(data_file))
        
        if self.dict_files[data_file]['labels'].shape[0] == 1:
            clip_list = np.reshape(np.array(list(clip.iter_frames())).mean(1), 
                                   (1, -1))
            
            return clip_list, self.dict_files[data_file]['labels']
        
        frames = []
        for i in range(8): #len(time) - 1):
            start_time = time[i]
            end_time = time[i + 1]
            data_frame = np.array(list(clip.subclip(start_time, end_time).iter_frames()))
            
            if 'audio' in self.input_type.lower():
                data_frame = data_frame.mean(1)[:num_samples]
                
            frames.append(data_frame)
        
        self.shape = data_frame.shape
        
        return frames, self.dict_files[data_file]['labels']
    
    def serialize_sample(self, writer, data_file, subject_id):
        
        for i, (frame, label) in enumerate(zip(*self._get_samples(data_file))):
            
            example = tf.train.Example(features=tf.train.Features(feature={
                        'sample_id': self._int_feauture(i),
                        'subject_id': self._bytes_feauture(subject_id.encode()),
                        'label': self._bytes_feauture(label.tobytes()),
                        'frame': self._bytes_feauture(frame.tobytes())
                    }))
            
            writer.write(example.SerializeToString())
            del frame, label
    