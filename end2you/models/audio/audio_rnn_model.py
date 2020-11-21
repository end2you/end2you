import torch
import torch.nn as nn
import numpy as np

from .audio_model import AudioModel
from end2you.models.rnn import RNN


class AudioRNNModel(nn.Module):
    
    def __init__(self, 
                 input_size:int, 
                 num_outs:int,
                 pretrained:bool = False,
                 model_name:str = None):
        ''' Audio model.
        
        Args:
            input_size (int): input size to the model. 
            num_outs (int): number of output values of the model.
        '''
        super(AudioRNNModel, self).__init__()
        audio_network = AudioModel(input_size)
        self.audio_model = audio_network.audio_model
        num_out_features = audio_network.num_features
        self.rnn, num_out_features = self._get_rnn_model(num_out_features)
        self.linear = nn.Linear(num_out_features, num_outs)
    
    def _get_rnn_model(self, input_size:int):
        '''Get RNN instace.'''
        rnn_args = {
            'input_size':input_size,
            'hidden_size':256,
            'num_layers':2
        }
        return RNN(rnn_args, 'lstm'), rnn_args['hidden_size']
    
    def forward(self, x:torch.Tensor):
        '''
        Args:
            x ((torch.Tensor) - BS x S x 1 x T)
        '''
        batch_size, seq_length, t = x.shape
        x = x.view(batch_size*seq_length, 1, t)
        
        audio_out = self.audio_model(x)        
        audio_out = audio_out.view(batch_size, seq_length, -1)
        
        rnn_out, (h_n, c_n) = self.rnn(audio_out)
        
        output = self.linear(rnn_out)
        return output
