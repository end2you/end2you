import h5py
import numpy as np
import sys
sys.path.append("..")

from pathlib import Path
from torch.utils.data import Dataset
from torch import is_tensor
from end2you.data_generator.file_reader import FileReader


class BaseProvider(Dataset):
    """Generic class. Audio/Visual providers inherit from this class."""
    
    def __init__(self,
                 input_file:str,
                 file_provider,
                 seq_length:int = None,
                 augment:bool = False,
                 fps:int = 16_000,
                 reader:FileReader = None,
                 *args, **kwargs):
        """ Initialize BaseProvider class.
        
        Args:
          dataset_path (list): List of data paths. 
          seq_length (int): Number of consequtive frames to read from video.
                            If `None` all video frames are provided.
                            (default `None`)
          augment (bool): Use data augmentation.
                          (default `False`)
        """
        if not reader:
            reader = FileReader()
        
        self.seq_length = seq_length if seq_length else 0
        self.raw_files, self.attr_names = reader.read_file(input_file)
        
        self.data_files = []
        for (raw_file, label_file) in self.raw_files:
            self.data_files.append(
                file_provider(raw_file, label_file, seq_length, reader, fps, *args, **kwargs))
        
        assert len(self.data_files) > 0, f'No files found in [{dataset_path}]. Check `dataset_path`.'

        self.num_files = len(self.data_files)
        self.total_num_seqs = np.ceil(self._get_total_num_seqs())
        self.label_names = self._get_label_names()
        
        self.augment = augment
        if augment:
            self.data_transform = self._init_augment()
    
    def _get_total_num_seqs(self):
        """ Get total number of sequences."""
        return sum([x.num_seqs for x in self.data_files])
    
    def _get_frame_num_samples(self, idx:int = 0):
        """ Get total number of frames in data file `idx`."""
        return self.data_files[idx].num_samples
    
    def _get_label_names(self, idx:int = 0):
        """ Get label names."""
        return self.data_files[idx]._get_label_names()
    
    def _init_augment(self):
        """ Identity method for augmenting data. 
            It is overloaded in sub-classes.
        """
        return lambda x: x
    
    def reset(self):
        """ Reset parameters to initial values."""
        for f in self.data_files:
            f.reset()
    
    def _get_file(self, idx):
        """ File to read data from next. 
        
        Args:
          idx (int): Index of file to read.
        """
        i = idx % self.num_files 
        file = self.data_files[i]
        while file.total_calls_reached():
            i = (i + 1) if (i+1) < self.num_files else 0
            file = self.data_files[i]
        
        return file
    
    def process_input(self, data, labels):
        """ Identity method for pre-process input. 
            It is overloaded in the sub-classes.
        """
        return data, labels
    
    def __len__(self):
        return sum([x.total_num_calls for x in self.data_files])
    
    def __getitem__(self, idx):
        data_file = self._get_file(idx)
        data, labels = data_file.read_file()
        data, labels = self.process_input(data, labels)
        return data, labels, str(data_file.raw_file_path)
