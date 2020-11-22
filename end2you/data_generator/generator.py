import numpy as np
import h5py
import logging

from pathlib import Path
from .file_reader import FileReader
from end2you.base_process import BaseProcess


class Generator:
    
    def __init__(self, 
                 save_data_folder: str,
                 reader:FileReader = None,
                 input_file:str = None,
                 *args, **kwargs):
        
        if reader:
            self.files, self.attr_names = reader.read_file(input_file, *args, **kwargs)
        self.save_data_folder = Path(save_data_folder)
        self.save_data_folder.mkdir(parents=True, exist_ok=True)
        BaseProcess.set_logger('generator.log')
        
    def write_data_files(self):
        
        logging.info('\n Start writing data files \n')
        
        for i, (data_file, label_file) in enumerate(self.files):
            data_file, label_file = Path(data_file), Path(label_file)
            logging.info('Writing .hdf5 file for : [{}]'.format(str(data_file)))
            
            file_name = self.save_data_folder / '{}.hdf5'.format(data_file.name[:-4])
            if file_name.exists():
                continue
            
            with h5py.File(str(file_name), 'w') as writer:
                self.serialize_samples(
                    writer, data_file, label_file)
    
    def serialize_samples(self, writer:h5py.File, data_file:str, label_file:str):
        raise NotImplementedError('Method not implemented!')
