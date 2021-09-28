from . import raw
from . import hdf5
from functools import partial

def get_proper_provider(files_format):
    return {
        'raw': get_raw_provider,
        'hdf5': get_hdf5_provider
    }[files_format]

def get_hdf5_provider(modality):
    """ Factory method to get the appropriate provider.
        
    Args:
      modality (str): Which modality provider to return.
    """
    
    return {
        'audio': hdf5.AudioProvider,
        'visual': hdf5.VisualProvider,
        'audiovisual': hdf5.SingleFile_AVProvider
    }[modality]

def get_raw_provider(modality):
    """ Factory method to get the appropriate provider.
    
    Args:
      modality (str): Which modality provider to return.
    """
    file_provider = _get_file_provider(modality)
    
    return {
        'audio': partial(raw.AudioProvider, file_provider=file_provider),
        'visual': partial(raw.VisualProvider, file_provider=file_provider),
#         'audiovisual': raw.SingleFile_AVProvider
    }[modality]


def _get_file_provider(modality):
    return {
        'audio': raw.AudioFileProvider,
        'visual': raw.VisualFileProvider,
#         'audiovisual': AVFileProvider
    }[modality]
