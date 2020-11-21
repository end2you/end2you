import numpy as np
import torch.nn as nn
import torch

from end2you.models.visual import VisualModel
from end2you.models.audio import AudioModel
from end2you.models.rnn import RNN


class AudioVisualRNNModel(nn.Module):
    
    def __init__(self, 
                 input_size:list, 
                 model_name:str,
                 num_outs:int,
                 pretrained:bool = False):
        ''' Audiovisual model.
        
        Args:
            input_size (list): List with the input size of both modalities.
                               First the audio size and then visual.
            model_name (str): Which visual network to use.
            num_outs (int): number of output values of the model.
        '''
        super(AudioVisualRNNModel, self).__init__()
        
        audio_num_samples = input_size[0]
        self.visual_model = VisualModel(model_name)
        num_visual_features = self.visual_model.num_features
        
        audio_network = AudioModel(audio_num_samples)
        self.audio_model = audio_network.audio_model
        num_audio_features = audio_network.num_features
        
        rnn_input_features = num_visual_features + num_audio_features
        self.rnn, num_out_features = self._get_rnn_model(rnn_input_features)
        self.linear = nn.Linear(num_out_features, num_outs)
    
    def _get_rnn_model(self, input_size:int):
        rnn_args = {
            'input_size':input_size,
            'hidden_size':512,
            'num_layers':2
        }
        return RNN(rnn_args, 'lstm'), rnn_args['hidden_size']
    
    def forward(self, model_input:list):
        '''
        Args:
            model_input (list): List with audio and visual tensor, respectively.
                visual_input (BS x S x C x H x W) : Visual tensor.
                audio_input (BS x S x 1 x T) : Audio tensor.
        Returns:
            Output of the model with dimension (BS x S x num_outs).
        '''
        audio_input, visual_input = model_input
        
        batch_size, seq_length, cv, h, w = visual_input.shape
        batch_size, seq_length, t = audio_input.shape
        
        audio_input = audio_input.view(batch_size * seq_length, 1, t)
        visual_input = visual_input.view(batch_size * seq_length, cv, h, w)
        
        audio_out = self.audio_model(audio_input)
        visual_out = self.visual_model(visual_input)
        
        audio_out = audio_out.view(batch_size, seq_length, -1)
        visual_out = visual_out.view(batch_size, seq_length, -1)
        
        multimodal_input = torch.cat([audio_out, visual_out], -1)
        
        rnn_out, (h_n, c_n) = self.rnn(multimodal_input)
        
        output = self.linear(rnn_out)
        return output.view(batch_size, seq_length, -1)
