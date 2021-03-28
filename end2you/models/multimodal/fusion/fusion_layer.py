import numpy as np
import torch.nn as nn
import torch

from .attention import AttentionFusion
from .concat import ConcatFusion
from functools import partial


class FusionLayer(nn.Module):
    """ Factory class to get fusion method. """
    
    def __init__(self,  
                 method:str,
                 num_feats_modality:list,
                 *args, **kwargs):
        """ Instantiate a fusion instance.
        
        Args:
            method (str): Fusion method to choose.
        """
        
        super(FusionLayer, self).__init__()
        self.fusion_layer = self._get_fusion(method, num_feats_modality)
        self.num_features = self.fusion_layer.num_features
    
    def _get_fusion(self, method:str, *args, **kwargs):
        """ Returns a fusion method instance.
        
        Args:
            method (str): Fusion method to return.
        """
        
        fusion_method = {
            'concat': ConcatFusion,
            'attention': AttentionFusion,
        }[method]
        
        fusion_instance = fusion_method(*args, **kwargs)
        return fusion_instance
    
    def forward(self, x:list):
        """ Forward pass
        
        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """
        return self.fusion_layer(x)