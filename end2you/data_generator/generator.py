import numpy as np
import h5py
import logging
import sys
sys.path.append("..") 

from pathlib import Path
from .file_reader import FileReader
from ..base_process import BaseProcess


class Generator:
    
    def __init__(self, 
                 save_data_folder: str,
                 reader:FileReader = None,
                 input_file:str = None,
                 *args, **kwargs):
        """ Initialize object class to generate `hdf5` files.
        
        Args:
          save_data_folder (str): Path to save `hdf5` files.
          reader (FileReader): Instance of class to read `input_file` file.
          input_file (str): File to read data/labels paths from.
        """
        
        if reader:
            self.files, self.attr_names = reader.read_file(input_file, *args, **kwargs)
        
        self.save_data_folder = Path(save_data_folder)
        self.save_data_folder.mkdir(parents=True, exist_ok=True)
        BaseProcess.set_logger('generator.log')
    
    def write_data_files(self):
        """ Main method that writes the `hdf5` files.
        """
        
        logging.info('\n Start writing data files \n')
        
        for i, (data_file, label_file) in enumerate(self.files):
            data_file, label_file = Path(data_file), Path(label_file)
            logging.info('Writing .hdf5 file for : [{}]'.format(str(data_file)))
            
            file_name = self.save_data_folder / '{}.hdf5'.format(label_file.name[:-4])
            if file_name.exists():
                continue
            
            with h5py.File(str(file_name), 'w') as writer:
                self.serialize_samples(
                    writer, data_file, label_file)
    
    def serialize_samples(self, writer:h5py.File, data_file:str, label_file:str):
        """ Base not implemented method to write data to `hdf5` file.
        
        Args:
          writer (h5py.File): Open file to write data.
          data_file (str): Data file name.
          label_file (str): Label file name.
        
        Throws:
          NotImplementedError Exception.
        """
        
        raise NotImplementedError('Method not implemented!')
