import numpy as np
import logging

from end2you.data_generator import get_generator, FileReader
from pathlib import Path
from end2you.utils import Params
from end2you.base_process import BaseProcess


class GenerationProcess(BaseProcess):
    """ Data generation process."""
    
    def __init__(self,
                 params:Params,
                 *args, **kwargs):
        
        modality_generator = get_generator(params.modality)
        
        fieldnames = params.fieldnames.split(',') if params.fieldnames else None
        
        if params.exclude_cols:
            exclude_cols = [
                int(x) for x in params.exclude_cols.split(',')]
        else:
            exclude_cols = []
        
        filereader = FileReader(',')
        labelfile_reader = FileReader(delimiter=params.delimiter, 
                              exclude_cols=exclude_cols, 
                              fieldnames=fieldnames)
        
        self.generator = modality_generator(save_data_folder=params.save_data_folder, 
                                 input_file=params.input_file,
                                 reader=filereader,
                                 labelfile_reader=labelfile_reader)
        
        log_file = Path(params.root_dir) / params.log_file
        self.set_logger(str(log_file))
        logging.info('Starting Generation Process')
    
    def start(self):
        self.generator.write_data_files()
