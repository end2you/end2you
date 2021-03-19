import csv
import numpy as np

from pathlib import Path


class FileReader:
    
    def __init__(self, 
                 delimiter:str = ',', 
                 exclude_cols:list = [],
                 fieldnames:list = None):
        """ Initialize object class to read a file.
        
        Args:
          delimiter (str): Delimiter used in the file (default `,`).
          exclude_cols (list): Exclude column in the file (default `None`).
          fieldnames (list): Names of the column in the file (default `None`).
        """
        
        self.delimiter = delimiter
        self.exclude_cols = exclude_cols
        self.fieldnames = fieldnames
    
    def read_file(self, file:str):
        """ Reads a file.
        
        Args:
          file (str): File path to read.
        
        Returns:
          data (np.array): The data contained in the file.
          attributes_name (list): List with the column names.
        """
        
        with open(file, 'r') as f:
            reader = csv.DictReader(f, 
                delimiter=self.delimiter, fieldnames=self.fieldnames)
            
            ncols = np.arange(len(reader.fieldnames))
            include_cols = np.delete(ncols, self.exclude_cols)
            reader_keys = [reader.fieldnames[x] for x in include_cols]
            
            data = []
            for row in reader:
                d = [row[x] for x in reader_keys]
                data.append(d)
        
        attributes_name = list(reader_keys)
        
        return data, attributes_name
