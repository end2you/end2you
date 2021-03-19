import torch
import torch.nn as nn
import numpy as np


class Base(nn.Module):
    """ Base class to build convolutional neural network model."""
    
    def __init__(self, 
                 conv_layers_args:dict,
                 maxpool_layers_args:dict,
                 conv_op:nn = nn.Conv1d,
                 max_pool_op:nn = nn.MaxPool1d,
                 activ_fn:nn = nn.LeakyReLU(),
                 normalize:bool = False):
        """ Audio model.
        
        Args:
            conv_layers_args (dict): parameters of convolutions layers.
            maxpool_layers_args (dict): parameters of max pool layer layers.
            conv_op (nn): convolution operation to use (default `nn.Conv1d`).
            max_pool_op (nn): max pooling operation to use (default `nn.MaxPool1d`).
            activ_fn (nn) : Activation function to use (default `nn.Relu()`).
            normalize (bool): Use batch normalization after convolution operation (default `False`).
        """
        
        super(Base, self).__init__()
        
        self.conv_op = conv_op
        network_layers = nn.ModuleList()
        for conv_args, mp_args in zip(*[conv_layers_args.values(), maxpool_layers_args.values()]):
            network_layers.extend([self._conv_block(conv_args, activ_fn, normalize)])
            network_layers.extend([max_pool_op(**mp_args)])
        
        self.network = nn.Sequential(*network_layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        """ Initialize parameters of the model."""
        for m in list(self.modules()):
            self._init_weights(m)
    
    def _init_weights(self, m):
        """ Helper method to initialize the parameters of the model 
            with Kaiming uniform initialization.
        """
        
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(param)
    
    @classmethod
    def _num_out_features(cls, input_size:int, conv_args:dict, mp_args:dict):
        """ Number of features extracted from Convolution Neural Network.
        
        Args:
            input_size (int): Number of samples of the frame.
            conv_args (dict): parameters of convolutions layers.
            mp_args (dict): parameters of max pool layer layers.
        """
        
        layer_input = input_size
        num_layers = len(conv_args)
        for i, (conv_arg, mp_arg) in enumerate(zip(*[conv_args.values(), mp_args.values()])):
            # number of features in the convolution output
            layer_input = np.floor(
                (layer_input - conv_arg['kernel_size'] + 2*conv_arg['padding']) / conv_arg['stride'] + 1)
            
            layer_input = np.floor(
                (layer_input - mp_arg['kernel_size']) / mp_arg['stride'] + 1)
        
        return int(layer_input)
    
    def _conv_block(self, conv_args:dict, activ_fn:nn, normalize:bool = False):
        """ Convolution block.
        
        Args:
            conv_args (dict): parameters of convolution layer.
            activ_fn (nn): Activation function to use (default `nn.Relu()`).
            normalize (bool): Use batch normalization after convolution 
                              operation (default `False`).
        """
        
        layer = nn.ModuleList([self.conv_op(**conv_args)])
        
        if normalize:
            layer.append(nn.BatchNorm1d(conv_args['out_channels']))
        
        layer.append(activ_fn)
        return nn.Sequential(*layer)
    
    def forward(self, x):
        """ Forwards pass.
        
        Args:
            x (BS x 1 x T)
        """
        return self.network(x)
