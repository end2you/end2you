from .audio_generator import AudioGenerator
from .visual_generator import VisualGenerator
from .audiovisual_generator import AudioVisualGenerator


def get_generator(modality):
    return {
        'audio': AudioGenerator,
        'visual': VisualGenerator,
        'audiovisual': AudioVisualGenerator
    }[modality]
