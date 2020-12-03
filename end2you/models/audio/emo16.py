import torch
import torch.nn as nn
import numpy as np


class Emo16(nn.Module):
    
    def __init__(self, 
                 input_size:int):
        ''' 
        Speech emotion recognition model proposed in:
        
        `Trigeorgis, G., Ringeval, F., Brueckner, R., Marchi, E., Nicolaou, M. A., Schuller, B., & Zafeiriou, S. 
        Adieu features? end-to-end speech emotion recognition using a deep convolutional recurrent network. 
        In 2016 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 5200-5204). IEEE.`
        
        Args:
            input_size (int): input size to the model. 
        '''
        super(Emo16, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=20, stride=1, padding=9)
        self.max_pool1 = nn.MaxPool1d(2, 2, padding=1)
        
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=40, stride=1, padding=19)
        self.max_pool2 = nn.MaxPool1d(10, 10, padding=4)
        
        self.num_features = int(np.ceil(input_size/2)) * 4
    
    def forward(self, x:torch.Tensor):
        '''
        Args:
            x ((torch.Tensor) - BS x S x 1 x T)
        '''
        batch_size, seq_length, t = x.shape
        x = x.view(batch_size*seq_length, 1, t)
        
        audio_out = self.conv1(x)        
        audio_out = self.max_pool1(audio_out)
        
        audio_out = self.conv2(audio_out)
        _, c2, t2 = audio_out.shape
        audio_out = audio_out.view(batch_size*seq_length, t2, c2)

        audio_out = self.max_pool2(audio_out)
        
        audio_out = audio_out.view(batch_size, seq_length, -1)
        
        return audio_out
