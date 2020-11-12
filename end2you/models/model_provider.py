import torch
import torch.nn as nn

from .audio_rnn_model import AudioRNNModel
from .visual_rnn_model import VisualRNNModel
from .audiovisual_rnn_model import AudioVisualRNNModel


def get_model(model:str = 'audio', *args, **kwargs):
    ''' Returns one of `audio`, `visual`, `audiovisual`.
    Args:
        model (str): Choice of model.
    '''
    return {
        'audio': AudioRNNModel,
        'visual': VisualRNNModel,
        'audiovisual': AudioVisualRNNModel
    }[model](*args, **kwargs)