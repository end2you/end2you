import numpy as np

from .provider import BaseProvider
from pathlib import Path
from torch import is_tensor


class AudioProvider(BaseProvider):
    """AudioProvider dataset."""
    
    def __init__(self, *args, **kwargs):
        self.modality = 'audio'
        super().__init__(*args, **kwargs)
    