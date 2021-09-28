import numpy as np

from pathlib import Path
from torch import is_tensor
from moviepy.editor import AudioFileClip
from .raw_file_provider import RawFileProvider


class AudioFileProvider(RawFileProvider):
    """Provides the data for the audio modality."""
    
    def __init__(self, *args, **kwargs):
        self.modality = 'audio'
        super().__init__(*args, **kwargs)
    
    def _get_fps(self):
        clip = AudioFileClip(str(self.raw_file_path))
        
        return clip.fps
    
    def read_file(self):
        
        start = self.num_calls*self.seq_length
        end = start + self.seq_length
            
        # Read label file, and get start-end timestamps
        labels, timestamps = self._read_label_file(start, end)
        
        clip = AudioFileClip(str(self.raw_file_path), fps=self.fps)
        
        num_seqs = self.num_seqs
        frames = []
        for i, t in enumerate(timestamps[:-1]):
            start_time = timestamps[i]
            end_time = timestamps[i+1]
            
            data_frame = np.array(
                list(clip.subclip(start_time, end_time).iter_frames()))
            data_frame = data_frame.mean(1)[:self.num_samples]
            
            if data_frame.shape[0] < self.num_samples:
                data_frame = np.pad(data_frame, 
                    (0, self.num_samples - data_frame.shape[0] % self.num_samples), 'constant')
            
            frames.append(data_frame.astype(np.float32))
        
        frames = np.array(frames).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        
        return frames, labels
