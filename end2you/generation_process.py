import numpy as np

from end2you.data_generator import get_generator, FileReader
from pathlib import Path
from end2you.utils import Params


class GenerationProcess:
    """AudioProvider dataset."""
    
    def __init__(self,
                 params:Params,
                 *args, **kwargs):
        
        modality_generator = get_generator(params.modality)
        
        filereader = FileReader(',')
        labelfile_reader = FileReader(delimiter=';', 
                              exclude_cols=[0], 
                              fieldnames=['file', 'timestamp', 'arousal', 'valence', 'liking'])
        
        self.generator = modality_generator(save_data_folder=params.save_data_folder, 
                                 input_file=params.input_file,
                                 reader=filereader,
                                 labelfile_reader=labelfile_reader)
        
    def start(self):
        self.generator.write_data_files()