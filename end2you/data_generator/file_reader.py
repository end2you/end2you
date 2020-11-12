import csv
import numpy as np

from pathlib import Path


class FileReader:
    
    def __init__(self, 
                 delimiter:str = ',', 
                 exclude_cols:list = [],
                 fieldnames:list = None):
        self.delimiter = delimiter
        self.exclude_cols = exclude_cols
        self.fieldnames = fieldnames
    
    def read_file(self, file:str):
        
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
