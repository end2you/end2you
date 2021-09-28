import h5py
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from .audio_provider import AudioProvider
from .visual_provider import VisualProvider


class MultiFile_AVProvider(Dataset):
    """Reads audiovisual data from two .hdf5 files."""
    
    def __init__(self,
                 dataset_paths:list,
                 seq_length:int = None):
        """ Initialize class.
            
        Args:
          dataset_paths (list): List of audio and visual paths. 
          seq_length (int): Number of consequtive frames to read from video. 
        """
        
        audio_dataset_path, visual_dataset_path = dataset_paths
        self.audio_provider = AudioProvider(audio_dataset_path, seq_length=seq_length)
        self.visual_provider = VisualProvider(visual_dataset_path, seq_length=seq_length)
    
    def _get_total_num_seqs(self):
        """Get total number of sequences."""
        return self.audio_provider._get_total_num_seqs()
    
    def _get_frame_num_samples(self, idx:int = 0):
        """Get total number of frames in the video `idx`."""
        return [self.audio_provider._get_frame_num_samples(idx),
               self.visual_provider._get_frame_num_samples(idx)]
    
    def _get_label_names(self):
        """Get label names."""
        return self.audio_provider._get_label_names()
    
    def reset(self):
        """Reset parameters to initial values."""
        self.audio_provider.reset()
        self.visual_provider.reset()
    
    def __len__(self):
        return len(self.audio_provider)
    
    def __getitem__(self, idx):
        audio_data, _ = self.audio_provider[idx]
        visual_data, labels = self.visual_provider[idx]
        return [audio_data, visual_data], labels 
