import torch
import torch.nn as nn
import numpy as np

from .base import Base


class Zhao19(nn.Module):
    
    def __init__(self, input_size:int):
        """ Speech emotion recognition model proposed in:
        
        `Zhao, J., Mao, X. and Chen, L.. "Speech emotion recognition using 
        deep 1D & 2D CNN LSTM networks". Biomedical Signal Processing and Control, 
        47, (pp.312-323), 2019.`
        
        Args:
            input_size (int): Input size to the model. 
        """
        
        super(Zhao19, self).__init__()
        self.model, self.num_features = self.build_audio_model(input_size)
    
    def build_audio_model(self, input_size:int):
        """ Build the audio model: 3 blocks of convolution + max-pooling.
        
        Args:
          input_size (int): Input size of frame.
        """
        
        out_channels = [64, 64, 128, 128]
        in_channels = [1]
        in_channels.extend([x for x in out_channels[:-1]])
        kernel_size = [3, 3, 3, 3]
        stride = [1, 1, 1, 1]
        padding = ((np.array(kernel_size)-1)//2).tolist()
        
        num_layers = len(in_channels)
        conv_args = {
            f'layer{i}':
                {
                    'in_channels': in_channels[i],
                    'out_channels': out_channels[i],
                    'kernel_size': kernel_size[i],
                    'stride': stride[i],
                    'padding': padding[i]
                }
            for i in range(num_layers)
         }
        
        kernel_size = [4, 4, 4, 4]
        stride = [4, 4, 4, 4]
        maxpool_args = {f'layer{i}': {
                        'kernel_size': kernel_size[i],
                        'stride': stride[i]
                    } for i in range(num_layers)
                 }
        
        audio_model = Base(conv_args, maxpool_args, normalize=True, activ_fn=nn.ELU())
        conv_red_size = Base._num_out_features(input_size, conv_args, maxpool_args)
        num_layers = len(in_channels) - 1
        num_out_features = conv_red_size*conv_args[f'layer{num_layers}']['out_channels']
        
        return audio_model, num_out_features
    
    def forward(self, x):
        '''
        Args:
            x (BS x 1 x T)
        '''
        return self.model(x)
