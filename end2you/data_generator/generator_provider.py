from .audio_generator import AudioGenerator
from .visual_generator import VisualGenerator
from .audiovisual_generator import AudioVisualGenerator


def get_generator(modality):
    """ Factory method to get the appropriate generator.
        
    Args:
      modality (str): Which modality provider to return.
    """
    
    return {
        'audio': AudioGenerator,
        'visual': VisualGenerator,
        'audiovisual': AudioVisualGenerator
    }[modality]
