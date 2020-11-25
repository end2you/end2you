import h5py
import numpy as np

from .file_provider import FileProvider
from pathlib import Path
from torch.utils.data import Dataset
from torch import is_tensor


class BaseProvider(Dataset):
    """Base class."""
    
    def __init__(self,
                 dataset_path:str,
                 seq_length:int = None):
        
        self.dataset_path = dataset_path
        self.seq_length = seq_length if seq_length else 0
        self.data_files = sorted(list(Path(dataset_path).glob('**/*.hdf5')))
        assert len(self.data_files) > 0, f'No files found in [{dataset_path}]. Check `dataset_path` flag.'
        self.data_files = [FileProvider(x, self.modality, seq_length) for x in self.data_files]
        self.num_files = len(self.data_files)
        self.total_num_seqs = np.ceil(self._get_total_num_seqs())
        self.label_names = self._get_label_names()
    
    def _get_total_num_seqs(self):
        ''' Get total number of sequences.'''
        return sum([x.num_seqs for x in self.data_files])
    
    def _get_frame_num_samples(self, idx:int = 0):
        ''' Get total number of sequences.'''
        return self.data_files[idx].num_samples
    
    def _get_label_names(self, idx:int = 0):
        ''' Get label names.'''
        return self.data_files[idx]._get_label_names()
    
    def reset(self):
        for f in self.data_files:
            f.reset()
    
    def _get_file(self, idx):
        i = idx % self.num_files 
        file = self.data_files[i]
        while file.total_calls_reached():
            i = (i + 1) if (i+1) < self.num_files else 0
            file = self.data_files[i]
        
        return file
    
    def process_input(self, data, labels):
        return data, labels
    
    def __len__(self):
        return sum([x.total_num_calls for x in self.data_files])    
    
    def __getitem__(self, idx):
        data_file = self._get_file(idx)
        data, labels = data_file.read_hdf5_file()
        data, labels = self.process_input(data, labels)
        return data, labels, data_file.file_path
    
