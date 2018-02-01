import csv
import numpy as np
import copy
import os
import arff

from pathlib import Path


class FileReader:
    
    def __init__(self, 
                 file:Path,
                 **kwargs):
        if not file:
            raise ValueError('Need to define a data file.')
        
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
                            exclude_cols:list = [],
                            delimiter='\t'):
        
        print('\n Start reading file [{}]\n'.format(file))
        with open(file, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            ncols = np.arange(len(reader.fieldnames))
            include_cols = np.delete(ncols, exclude_cols)
            reader_keys = [reader.fieldnames[x] for x in include_cols]
            
            data = []
            for row in reader:
                d = [row[x] for x in reader_keys]
                data.append(d)
        
        keys = list(reader_keys)
        attributes_name = copy.deepcopy(keys)
        file_idx = attributes_name.index('file')
        
        attributes_type = ['str']
        if not os.path.isfile(keys[0]):
            attributes_type = []
            for i, k in enumerate(keys):
                if i == file_idx:
                    continue
                try:
                    int(data[0][i])
                    attributes_type.append('int')
                except ValueError:
                    try:
                        float(keys[i])
                        attributes_type.append('float')
                    except ValueError:
                        raise ValueError('Only integers and floats are supported for the label.')
            attributes_type.insert(file_idx, 'str')
        
        return attributes_name, attributes_type, np.array(data)
    
    @classmethod
    def read_arff_file(cls, 
                       file):
        
        data = arff.load(open(file, 'r'))
        attributes_name, attributes_type = list(zip(*data_arff["attributes"]))
        data = data_arff["data"]
        
        return attributes_name, attributes_type, np.array(data)
