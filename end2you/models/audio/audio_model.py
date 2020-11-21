import torch
import torch.nn as nn

from .emo16 import Emo16
from .emo18 import Emo18


class AudioModel(nn.Module):
    
    def __init__(self, 
                 model_name:str,
                 pretrained:bool = False,
                 *args, **kwargs):
        ''' Network model.
        
        Args:
            model_name: Name of audio model to use.
        '''
        super(AudioModel, self).__init__()
        
        self.model = self._get_model(model_name)
        self.model = self.model(*args, **kwargs)
        self.num_features = self.model.num_features
        
    def _get_model(self, model_name):
        return {
            'emo16': Emo16,
            'emo18': Emo18
        }[model_name]
    
    def forward(self, x):
        '''
        Args:
            x (BS x S x 1 x T)
        '''
        return self.model(x)
