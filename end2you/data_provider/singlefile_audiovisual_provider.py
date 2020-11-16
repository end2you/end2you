import h5py
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from .audio_provider import AudioProvider
from .visual_provider import VisualProvider
from .provider import BaseProvider


class SingleFile_AVProvider(BaseProvider):
    """Base class."""
    
    def __init__(self,
                 *args, **kwargs):
        self.modality = ['audio', 'visual']
        super().__init__(*args, **kwargs)
    
    def process_input(self, data, labels):
        return [data[0], data[1]/255.], labels
    
