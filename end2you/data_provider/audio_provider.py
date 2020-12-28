import numpy as np

from .provider import BaseProvider
from pathlib import Path
from torch import is_tensor


class AudioProvider(BaseProvider):
    """AudioProvider dataset."""
    
    def __init__(self, *args, **kwargs):
        self.modality = 'audio'
        super().__init__(*args, **kwargs)
    
    def process_input(self, data, labels):
#         max_val = np.max(np.abs(data), 1).reshape(-1, 1)
#         mean_val = np.mean(data, 1).reshape(-1, 1)
#         std_val = np.std(data, 1).reshape(-1, 1)
        
#         return (data - mean_val) / std, labels
        return data, labels
    