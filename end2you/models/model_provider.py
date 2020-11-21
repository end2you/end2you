import torch
import torch.nn as nn

from .audio import AudioRNNModel
from .visual import VisualRNNModel
from .multimodal import AudioVisualRNNModel


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
