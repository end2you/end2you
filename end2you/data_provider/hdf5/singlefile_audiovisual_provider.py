import h5py
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from .audio_provider import AudioProvider
from .visual_provider import VisualProvider
from .base_provider import BaseProvider


class SingleFile_AVProvider(BaseProvider):
    """ Reads audiovisual data from a single .hdf5 file."""
    
    def __init__(self,
                 *args, **kwargs):
        """ Initialize class.
            
        Args:
          dataset_path (list): List of data paths. 
          seq_length (int): Number of consequtive frames to read from video.
                            If `None` all video frames are provided.
                            (default `None`)
          augment (bool): Use data augmentation.
                          (default `False`)
        """
        
        self.modality = ['audio', 'visual']
        super().__init__(*args, **kwargs)
    
    def process_input(self, data, labels):
        """ Pre-process input frames/labels.
            Can be used for augmentation.
            
        Args:
          data (list): List of audio and visual frames. 
          labels (np.array): Labels. 
        """
        return [data[0], data[1][:,0,...]/255.], labels
    
