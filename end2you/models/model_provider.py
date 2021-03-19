import torch
import torch.nn as nn

from .audio import AudioRNNModel
from .visual import VisualRNNModel
from .multimodal import AudioVisualRNNModel


def get_model(model:str = 'audio', *args, **kwargs):
    """ Factory method to provide a model of choice 
        (`audio`, `visual`, `audiovisual`).
    
    Args:
        model (str): Model to use.
    """
    
    return {
        'audio': AudioRNNModel,
        'visual': VisualRNNModel,
        'audiovisual': AudioVisualRNNModel
    }[model](*args, **kwargs)
