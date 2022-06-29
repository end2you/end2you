import numpy as np

from csv import reader
from end2you.data_generator.file_reader import FileReader


class RawFileProvider:
    """Helper class to read hdf5 files."""
    
    def __init__(self,
                 raw_file_path:str,
                 label_file_path:str,
                 seq_length:int = None,
                 reader:FileReader = None,
                 fps:int = None,
                 *args, **kwargs):
        """ Initialize class object.
        
        Args:
          file_path (str): Path to label `csv` file to read.
          modality (str): Modality to read data.
          seq_length (int): Number of consecutive frames to provide.
        """
        
        self.raw_file_path = raw_file_path
        self.label_file_path = label_file_path
        
        self.num_calls = 0
        self.fps = fps if fps else self._get_fps()
        self.num_samples = self._get_num_samples()
        self.num_seqs = self._get_num_sequences()
        self.seq_length = self.num_seqs if seq_length == None else seq_length
        self.total_num_calls = np.ceil(self.num_seqs / self.seq_length).astype(int)
    
    def _get_num_samples(self):
        """ Returns the number of samples in the file.
        """
        
        timestamps = []
        for i, row in enumerate(open(self.label_file_path, 'r')):
            if i == 0:
                continue
            
            timestamps.append(float(row.split(',')[0]))
            if i == 2:
                break
        
        num_samples = int(self.fps * (timestamps[1] - timestamps[0]))
        
        return num_samples
    
    def _get_num_sequences(self):
        """ Returns the number of sequences in the file.
            Stored in file as attribute with name `seq_num`.
        """
        
        num_seqs = -1
        for _ in open(self.label_file_path, 'r'):
            num_seqs += 1
        
        return num_seqs
    
    def _get_label_names(self):
        """ Returns the names of the labels.
            Stored in file as attribute with name `label_names`.
        """
        
        for i, row in enumerate(open(self.label_file_path, 'r')):
            label_names = row[1:]
            break
        
        return label_names
    
    def reset(self):
        """ Reset the parameter needed to re-read the data.
            Performed Uusually after each epoch.
        """
        self.num_calls = 0
    
    def total_calls_reached(self):
        """ Checks whether all data were read."""
        return self.total_num_calls == self.num_calls
    
    def _read_label_file(self, start, end):
        """ Read label file from `start` to `end`."""
        
        infos = np.loadtxt(self.label_file_path, skiprows=1, dtype=float, delimiter=',')
        labels = infos[start:end, 1:].astype(np.float32)
        timestamps = infos[start:(end+1), 0].astype(np.float32)
        
        return labels, timestamps

    def read_file(self):
        raise RuntimeError('Method not implemented!')
