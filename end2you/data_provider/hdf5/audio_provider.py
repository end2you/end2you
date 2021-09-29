import numpy as np

from .base_provider import BaseProvider
from pathlib import Path
from torch import is_tensor


class AudioProvider(BaseProvider):
    """Provides the data for the audio modality."""
    
    def __init__(self, *args, **kwargs):
        self.modality = 'audio'
        super().__init__(*args, **kwargs)
    
    def process_input(self, data, labels):
        return data, labels
    