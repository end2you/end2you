import torch
import torch.nn as nn


class RNN(nn.Module):
    
    def __init__(self, 
                 arch_args:dict,
                 arch:str = 'lstm'):
        '''RNN model.
        
        Args:
            arch_args (dict): arguments of the architecture.
               keys: input_size, hidden_size, num_layers
            arch (str): lstm or gru
        '''
        super(RNN, self).__init__()
        self.rnn = self._get_rnn(arch, arch_args)
    
    def _get_rnn(self, arch, arch_args):
        return {
            'lstm':nn.LSTM,
            'gru':nn.GRU
        }[arch](**arch_args)
    
    def forward(self, x):
        '''
        Args:
            x (BS x C x T)
        '''
        return self.rnn(x)
    