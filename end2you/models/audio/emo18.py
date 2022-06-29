import torch
import torch.nn as nn
import numpy as np

from .base import Base


class Emo18(nn.Module):
    
    def __init__(self, input_size:int):
        """ Speech emotion recognition model proposed in:
        
        `Tzirakis, P., Zhang, J., and Schuller, BW. "End-to-end speech emotion recognition 
        using deep neural networks." In 2018 IEEE international conference on acoustics, speech 
        and signal processing (ICASSP), (pp. 5089-5093). IEEE.`
        
        Args:
            input_size (int): Input size to the model. 
        """
        
        super(Emo18, self).__init__()
        self.model, self.num_features = self.build_audio_model(input_size)
    
    def build_audio_model(self, input_size:int):
        """ Build the audio model: 3 blocks of convolution + max-pooling.
        
        Args:
          input_size (int): Input size of frame.
        """
        
        out_channels = [64, 128, 256]
        in_channels = [1]
        in_channels.extend([x for x in out_channels[:-1]])
        kernel_size = [8, 6, 6]
        stride = [1, 1, 1]
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
        
        kernel_size = [10, 8, 8]
        stride = [10, 8, 8]
        maxpool_args = {f'layer{i}': {
                        'kernel_size': kernel_size[i],
                        'stride': stride[i]
                    } for i in range(num_layers)
                 }
        
        audio_model = Base(conv_args, maxpool_args, normalize=True)
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
