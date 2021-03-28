import numpy as np
import torch.nn as nn
import torch

from end2you.models.visual import VisualModel
from end2you.models.audio import AudioModel
from end2you.models.rnn import RNN
from end2you.models.multimodal.fusion import FusionLayer


class AudioVisualRNNModel(nn.Module):
    
    def __init__(self, 
                 input_size:list, 
                 num_outs:int,
                 model_name:list = ['emo16', 'resnet18'],
                 fusion_method:str = 'concat',
                 pretrained:bool = False,
                 *args, **kwargs):
        """ Audiovisual model.
        
        Args:
            input_size (list): List with the input size of both modalities.
                               First the audio size and then visual.
            model_name (str): Which visual network to use.
            num_outs (int): Number of output values of the model.
            pretrained (bool): Whether to use pretrain model (default `False`).
        """
        
        super(AudioVisualRNNModel, self).__init__()
        
        # Initialize Visual model
        audio_num_samples = input_size[0]
        self.visual_model = VisualModel(model_name[1])
        num_visual_features = self.visual_model.num_features
        
        # Initialize Audio model
        self.audio_model = AudioModel(model_name[0], input_size=audio_num_samples)
        num_audio_features = self.audio_model.num_features
        
        # Initialize Fusion layer
        self.fusion_layer = FusionLayer(fusion_method, 
                                        num_feats_modality=[num_audio_features, num_visual_features]) 
        
        # Initialize RNN layer
        rnn_input_features = self.fusion_layer.num_features
        self.rnn, num_out_features = self._get_rnn_model(rnn_input_features)
        self.linear = nn.Linear(num_out_features, num_outs)
    
    def _get_rnn_model(self, input_size:int):
        """ Builder method to get RNN instace."""
        
        rnn_args = {
            'input_size':input_size,
            'hidden_size':512,
            'num_layers':2
        }
        return RNN(rnn_args, 'lstm'), rnn_args['hidden_size']
    
    def forward(self, model_input:list):
        """ Forward pass.
        
        Args:
            model_input (list): List with audio and visual tensor, respectively.
                visual_input (BS x S x C x H x W) : Visual tensor.
                audio_input (BS x S x 1 x T) : Audio tensor.
        Returns:
            Output of the model with dimension (BS x S x num_outs).
        """
        
        audio_input, visual_input = model_input
        
        batch_size, seq_length, cv, h, w = visual_input.shape
        batch_size, seq_length, t = audio_input.shape
        
        audio_input = audio_input.view(batch_size * seq_length, 1, t)
        visual_input = visual_input.view(batch_size * seq_length, cv, h, w)
        
        audio_out = self.audio_model(audio_input)
        visual_out = self.visual_model(visual_input)
        
        audio_out = audio_out.view(batch_size, seq_length, -1)
        visual_out = visual_out.view(batch_size, seq_length, -1)
        
        multimodal_input = self.fusion_layer([audio_out, visual_out])
        
        rnn_out, (h_n, c_n) = self.rnn(multimodal_input)
        
        output = self.linear(rnn_out)
        return output.view(batch_size, seq_length, -1)