import h5py
import numpy as np


class FileProvider:
    """Helper class to read hdf5 files."""
    
    def __init__(self,
                 file_path:str,
                 modality:str,
                 seq_length:int = None):
        """ Initialize class object.
        
        Args:
          file_path (str): Path to `hdf5` file to read.
          modality (str): Modality to read data.
          seq_length (int): Number of consecutive frames to provide.
        """
        
        self.modality = modality
        self.file_path = file_path
        self.num_calls = 0
        self.num_samples = self._get_num_samples()
        self.num_seqs = self._get_num_sequences()
        self.seq_length = self.num_seqs if seq_length == None else seq_length
        self.total_num_calls = np.ceil(self.num_seqs / self.seq_length).astype(int)
    
    def _get_num_samples(self, key:str = 'num_samples'):
        """ Returns the number of samples in the file.
            Stored in file as attribute with name `num_samples`.
        """
        with h5py.File(self.file_path, 'r') as dataset:
            num_samples = dataset.attrs[key]
        
        return num_samples
    
    def _get_num_sequences(self, key:str = 'seq_num'):
        """ Returns the number of sequences in the file.
            Stored in file as attribute with name `seq_num`.
        """

        with h5py.File(self.file_path, 'r') as dataset:
            num_samples = dataset.attrs[key]
        
        return num_samples
    
    def _get_label_names(self, key:str = 'label_names'):
        """ Returns the names of the labels.
            Stored in file as attribute with name `label_names`.
        """

        with h5py.File(self.file_path, 'r') as dataset:
            label_names = dataset.attrs[key]
        
        return label_names
    
    def reset(self):
        """ Reset the parameter needed to re-read the data.
            Performed Uusually after each epoch.
        """
        self.num_calls = 0
    
    def total_calls_reached(self):
        """ Checks whether all data were read."""
        return self.total_num_calls == self.num_calls
    
    def __get_data(self, dataset:h5py.File, start:int, end:int):
        """ Reads and returns the requested data.
            
        Args:
          dataset (h5py.File): File to get data from.
          start (int): Start frame of sequence.
          end (int): End frame of sequence.
        """
        
        if isinstance(self.modality, list):
            return [np.array(dataset[x][start:end]) for x in self.modality]
        else:
            return np.array(dataset[self.modality][start:end])
    
    def read_hdf5_file(self):
        """ Read `hdf5` file and return data and labels.
        
        Returns:
          data (np.array): Data array.
          labels (np.array): Labels array.
        """
        
        data, labels = [], []
        with h5py.File(self.file_path, 'r') as dataset:
            
            start = self.num_calls*self.seq_length # if self.num_calls*self.seq_length < self.num_seqs else 0
            end = start + self.seq_length
            
            data = self.__get_data(dataset, start, end)
            labels = np.array(dataset['labels'][start:end])
        
        self.num_calls += 1
        return data, labels
