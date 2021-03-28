import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """ Fuse modalities using attention. """
    
    def __init__(self,  
                 num_feats_modality:list):
        """ Instantiate attention fusion instance.
        
        Args:
            num_feats_modality (list): Number of features per modality.
        """
        
        super(ConcatFusion, self).__init__()
        self.num_features = sum([x for x in num_feats_modality])
    
    def forward(self, x:list):
        """ Forward pass
        
        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """
        
        return torch.cat(x, dim=-1)
        
