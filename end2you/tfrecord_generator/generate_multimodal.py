import os
import tensorflow as tf
import numpy as np

from .generator import Generator
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path


class MultimodalGenerator(Generator):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_samples(self, data_file):
        
        time = self.dict_files[data_file]['time']

        clip = VideoFileClip(str(data_file))
        
        time = time[1:10,:]
        subsampled_audio = clip.audio.set_fps(16000)
        num_samples = int(subsampled_audio.fps * (time[1] - time[0]))
        
        audio_frames = []
        video_frames = []

        for i in range(len(time) - 1):
            start_time = time[i]
            end_time = time[i + 1]
            audio = np.array(list(subsampled_audio.subclip(start_time, end_time).iter_frames()))
            audio = audio.mean(1)[:num_samples]
                
            audio_frames.append(audio.astype(np.float32))
            
            image = np.array(list(clip.subclip(start_time, end_time).iter_frames()))[0]
            video_frames.append(image.astype(np.float32))
            
        return video_frames, audio_frames, self.dict_files[data_file]['labels']

    def serialize_sample(self, writer, data_file, subject_id):
        
        for i, (frame, audio, label) in enumerate(zip(*self._get_samples(data_file))):
            frame_shape = np.array(frame.shape)
            label_shape = np.array(label.shape)
            raw_audio_shape = np.array(audio.shape)
            
            example = tf.train.Example(features=tf.train.Features(feature={
                        'sample_id': self._int_feauture(i),
                        'subject_id': self._bytes_feauture(subject_id.encode()),
                        'label': self._bytes_feauture(label.tobytes()),
                        'label_shape': self._bytes_feauture(label_shape.tobytes()),
                        'raw_audio': self._bytes_feauture(audio.tobytes()),
                        'raw_audio_shape': self._bytes_feauture(raw_audio_shape.tobytes()),
                        'frame': self._bytes_feauture(frame.tobytes()),
                        'frame_shape': self._bytes_feauture(frame_shape.tobytes())
                    }))

            writer.write(example.SerializeToString())
            del frame, audio, label
    