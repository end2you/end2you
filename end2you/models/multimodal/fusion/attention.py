import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """ Fuse modalities using attention. """
    
    def __init__(self,  
                 num_feats_modality:list,
                 num_out_feats:int = 256):
        """ Instantiate attention fusion instance.
        
        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
        """
        
        super(AttentionFusion, self).__init__()
        
        self.attn = nn.ModuleList([])
        for num_feats in num_feats_modality:
            self.attn.append(
                nn.Linear(num_feats, num_out_feats))
        
        self.weights = nn.Linear(num_out_feats*2, num_out_feats*2)
        self.num_features = num_out_feats*2
    
    def forward(self, x:list):
        """ Forward pass
        
        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """
        
        proj_m = []
        for i, m in enumerate(x):
            proj_m.append(self.attn[i](m))
        
        attn_weights = F.softmax(
            self.weights(torch.cat(proj_m, -1)), dim=-1)
        
        out_feats = attn_weights * torch.cat(proj_m, -1)
        
        return out_feats