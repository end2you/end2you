import torch
import torch.nn as nn
import numpy as np

from .visual_model import VisualModel
from end2you.models.rnn import RNN


class VisualRNNModel(nn.Module):
    
    def __init__(self, 
                 model_name:str,
                 input_size:int = 1, 
                 pretrained:bool = False,
                 num_outs:int = 2):
        """ Visual model with an RNN on top.
        
        Args:
            model_name (str): Which visual model to use.
            input_size (int): input size to the model (default `1`). 
            pretrained (bool): Use pretrained (on ImageNet) model (default `False`).
            num_outs (int): number of output values of the model (default `2`).
        """
        
        super(VisualRNNModel, self).__init__()
        
        self.visual_model = VisualModel(model_name, pretrained)
        num_out_features = self.visual_model.num_features
        
        self.rnn, num_out_features = self._get_rnn_model(num_out_features)
        self.linear = nn.Linear(num_out_features, num_outs)
        self.num_outs = num_outs
        
    def _get_rnn_model(self, input_size:int):
        """ Builder method to instantiate an RNN object."""
        
        rnn_args = {
            'input_size': input_size,
            'hidden_size': 256,
            'num_layers': 2,
            'batch_first': True
        }
        return RNN(rnn_args, 'gru'), rnn_args['hidden_size']
    
    def forward(self, x):
        """ Forward pass.
        
        Args:
            x (BS x S x C x H x W)
        """
        
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size*seq_length, c, h, w)
        
        visual_out = self.visual_model(x)
        
        visual_out = visual_out.view(batch_size, seq_length, -1)
        
        rnn_out, _ = self.rnn(visual_out)
        
        output = self.linear(rnn_out)
        
        return output
