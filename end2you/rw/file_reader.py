import csv
import numpy as np

from pathlib import Path


class FileReader:
    
    def __init__(self, 
                 file:Path,
                 **kwargs):
        
        self.file = str(file)
        self.type = self.file.split('.')[-1][-2:]
        self.kwargs = kwargs
    
    def read(self):
        return {'ff': self.read_arff_file,
                'sv': self.read_delimiter_file}[self.type](self.file, 
                                                         **self.kwargs)
    
    @classmethod
    def read_delimiter_file(cls,
                            file, 
                            delimiter=';'):

        with open(file, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            data = []
            for row in reader:
                data.append(list(row.values()))

        keys = list(row.keys())
        attributes = [keys[x].split('@') for x in range(len(keys))]
        attributes_name = [x[0] for x in attributes]
        attributes_type = [x[1] for x in attributes]
        
        return attributes_name, attributes_type, np.array(data)
    
    @classmethod
    def read_arff_file(cls, 
                       file):
        
        data = arff.load(open(file, 'r'))
        attributes_name, attributes_type = list(zip(*data_arff["attributes"]))
        data = data_arff["data"]
        
        return attributes_name, attributes_type, np.array(data)