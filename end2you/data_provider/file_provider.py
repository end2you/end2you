import h5py
import numpy as np


class FileProvider:
    
    def __init__(self,
                 file_path:str,
                 modality:str,
                 seq_length:int = None):
        
        self.modality = modality
        self.file_path = file_path
        self.num_calls = 0
        self.num_samples = self._get_num_samples()
        self.num_seqs = self._get_num_sequences()
        self.seq_length = self.num_seqs if seq_length == None else seq_length
        self.total_num_calls = np.ceil(self.num_seqs / self.seq_length).astype(int)
    
    def _get_num_samples(self, key:str = 'num_samples'):
        with h5py.File(self.file_path, 'r') as dataset:
            num_samples = dataset.attrs[key]
        
        return num_samples
    
    def _get_num_sequences(self, key:str = 'seq_num'):
        with h5py.File(self.file_path, 'r') as dataset:
            num_samples = dataset.attrs[key]
        
        return num_samples
    
    def _get_label_names(self, key:str = 'label_names'):
        with h5py.File(self.file_path, 'r') as dataset:
            label_names = dataset.attrs[key]
        
        return label_names
    
    def reset(self):
        self.num_calls = 0
    
    def total_calls_reached(self):
        return self.total_num_calls == self.num_calls
    
    def __get_data(self, dataset:h5py.File, start:int, end:int):
        if isinstance(self.modality, list):
            return [np.array(dataset[x][start:end]) for x in self.modality]
        else:
            return np.array(dataset[self.modality][start:end])
    
    def read_hdf5_file(self):
        '''Returns a numpy array of the data.'''
        
        data, labels = [], []
        with h5py.File(self.file_path, 'r') as dataset:
            
            start = self.num_calls*self.seq_length # if self.num_calls*self.seq_length < self.num_seqs else 0
            end = start + self.seq_length
            
            data = self.__get_data(dataset, start, end)
            labels = np.array(dataset['labels'][start:end])
        
        self.num_calls += 1
        return data, labels
