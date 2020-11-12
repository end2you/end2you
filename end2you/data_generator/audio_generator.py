import numpy as np
import h5py

from pathlib import Path
from moviepy.editor import AudioFileClip
from .generator import Generator
from .file_reader import FileReader


class AudioGenerator(Generator):
    
    def __init__(self, 
                 labelfile_reader:FileReader, 
                 fps:int = 16000, 
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.fps = fps
        self.labelfile_reader = labelfile_reader
        
    def _get_samples(self, data_file, label_file):
        
        file_data, attrs_name = self.labelfile_reader.read_file(label_file)
        file_data = np.array(file_data).astype(np.float32)
        timestamps = file_data[:,0]
        labels = file_data[:-1,1:]
        
        seq_num = labels.shape[0] - 1
        
        clip = AudioFileClip(str(data_file), fps=self.fps)
        
        num_samples = int(self.fps * (timestamps[1] - timestamps[0]))
        frames = []        
        for i in range(len(timestamps) - 1):
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            
#             try: 
            data_frame = np.array(list(clip.subclip(start_time, end_time).iter_frames()))
            data_frame = data_frame.mean(1)[:num_samples]
            
            frames.append(data_frame.astype(np.float32))
#             except Exception as E:
#                 print(f'Exception during generating file: [{E}]')
#                 print('Continuing extracting files')
#                 labels = np.delete(labels, i, 0)
        
        frames = np.array(frames).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        
        return frames, labels, seq_num, num_samples, attrs_name
    
    def serialize_samples(self, writer, data_file, label_file):
        frames, labels, seq_num, num_samples, names = self._get_samples(data_file, label_file)
        
        # store data
        writer.create_dataset('audio', data=frames)
        writer.create_dataset('labels', data=labels)
        
        # Save meta-data
        writer.attrs['data_file'] = str(data_file)
        writer.attrs['label_file'] = str(label_file)
        writer.attrs['seq_num'] = seq_num
        writer.attrs['num_samples'] = num_samples ####### ******* ########
        writer.attrs['label_names'] = names[1:]
        