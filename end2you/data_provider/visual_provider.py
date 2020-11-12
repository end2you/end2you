import numpy as np

from pathlib import Path
from torch import is_tensor
from .provider import BaseProvider


class VisualProvider(BaseProvider):
    """AudioProvider dataset."""
    
    def __init__(self, *args, **kwargs):
        self.modality = 'visual'
        super().__init__(*args, **kwargs)
    
    def process_input(self, data, labels):
        return data[:,0,...]/255., labels
    
#     def __getitem__(self, idx):
#         if is_tensor(idx):
#             idx = idx.tolist()
        
#         data, labels, num_samples_masked = self.data_files[idx].read_hdf5_file(
#             self.seq_length)
        
#         return data, labels, num_samples_masked