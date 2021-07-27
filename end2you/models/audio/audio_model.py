import torch
import torch.nn as nn

from .emo16 import Emo16
from .emo18 import Emo18
from .zhao19 import Zhao19


class AudioModel(nn.Module):
    
    def __init__(self, 
                 model_name:str,
                 pretrained:bool = False,
                 *args, **kwargs):
        """ Audio network model.
        
        Args:
            model_name (str): Name of audio model to use.
            pretrain (bool): Whether to use pretrain model (default `False`).
        """
        
        super(AudioModel, self).__init__()
        
        self.model = self._get_model(model_name)
        self.model = self.model(*args, **kwargs)
        self.num_features = self.model.num_features
        
    def _get_model(self, model_name):
        """ Factory method to choose audio model."""
        
        return {
            'emo16': Emo16,
            'emo18': Emo18,
            'zhao19': Zhao19
        }[model_name]
    
    def forward(self, x):
        """ Forward pass
        
        Args:
            x (BS x S x 1 x T)
        """
        return self.model(x)
