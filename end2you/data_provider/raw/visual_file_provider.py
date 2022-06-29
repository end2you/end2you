import numpy as np
import sys
sys.path.append("..")

from pathlib import Path
from torch import is_tensor
from moviepy.editor import VideoFileClip
from .raw_file_provider import RawFileProvider
from end2you.data_generator.face_extractor import FaceExtractor


class VisualFileProvider(RawFileProvider):
    """Provides the data for the audio modality."""
    
    def __init__(self, *args, **kwargs):
        self.modality = 'visual'
        self.detector = FaceExtractor()
        super().__init__(*args, **kwargs)
    
    def _get_fps(self):
        clip = VideoFileClip(str(self.raw_file_path))
        
        return clip.fps
    
    def read_file(self):
        
        start = self.num_calls*self.seq_length
        end = start + self.seq_length
            
        # Read label file, and get start-end timestamps
        labels, timestamps = self._read_label_file(start, end)
        
        clip = VideoFileClip(str(self.raw_file_path))
        
        frames = []        
        for i in range(len(timestamps) - 1):
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            
            data_frame = np.array(list(clip.subclip(start_time, end_time).iter_frames()))
            data_frame = data_frame[:self.num_samples]
            
            if self.detector:
                try:
                    data_frame = self.detector.extract_and_resize_face(data_frame)
                except:
                    data_frame = np.zeros(
                        (1, self.detector.resize[0], self.detector.resize[0], 3), 
                        dtype=np.float32)
            
            if data_frame.shape[0] < self.num_samples:
                data_frame = np.pad(data_frame, (
                        (0,self.num_samples - data_frame.shape[0] % self.num_samples),
                        (0,0),(0,0),(0,0)), 'reflect')
            
            data_frame = data_frame.transpose(0, 3, 1, 2)
            
            frames.append(data_frame.astype(np.float32))
        
        frames = np.array(frames).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        
        return frames, labels
