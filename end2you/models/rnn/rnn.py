import torch
import torch.nn as nn


class RNN(nn.Module):
    
    def __init__(self, 
                 arch_args:dict,
                 arch:str = 'lstm'):
        """ Initialize RNN instance.
        
        Args:
            arch_args (dict): arguments of the architecture.
               keys: input_size, hidden_size, num_layers
            arch (str): lstm or gru (default `lstm`).
        """
        
        super(RNN, self).__init__()
        self.rnn = self._get_rnn(arch, arch_args)
    
    def _get_rnn(self, arch, arch_args):
        """ Factory method to get RNN instance."""
        
        return {
            'lstm':nn.LSTM,
            'gru':nn.GRU
        }[arch](**arch_args)
    
    def forward(self, x):
        """ Forward pass.
        
        Args:
            x (BS x C x T)
        """
        
        self.rnn.flatten_parameters()
        return self.rnn(x)
    