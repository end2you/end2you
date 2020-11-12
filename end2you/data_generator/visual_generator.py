import torch
import numpy as np
import h5py

from pathlib import Path
from moviepy.editor import VideoFileClip

from .generator import Generator
from .file_reader import FileReader
from .face_extractor import FaceExtractor


class VisualGenerator(Generator):
    
    def __init__(self,
                 labelfile_reader:FileReader, 
                 detector:FaceExtractor = FaceExtractor(),
                 fps:int = 30, 
                 *args, **kwargs):
        
        self.labelfile_reader = labelfile_reader
        self.fps = fps
        self.detector = detector
        super().__init__(*args, **kwargs)
    
    def _get_samples(self, data_file, label_file):
        
        file_data, attrs_name = self.labelfile_reader.read_file(label_file)
        file_data = np.array(file_data).astype(np.float32)
        timestamps = file_data[:,0]
        labels = file_data[:-1,1:]
        
        clip = VideoFileClip(str(data_file))
        
        seq_num = labels.shape[0] - 1
        num_samples = int(self.fps * (timestamps[1] - timestamps[0]))
        
        frames = []        
        for i in range(len(timestamps) - 1):
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            
            data_frame = np.array(list(clip.subclip(start_time, end_time).iter_frames()))
            data_frame = data_frame[:num_samples]
            
            if self.detector:
                data_frame = self.detector.extract_and_resize_face(data_frame)
            data_frame = data_frame.transpose(0, 3, 1, 2)
            
            frames.append(data_frame.astype(np.float32))
        
        frames = np.array(frames).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        
        return frames, labels, seq_num, num_samples, attrs_name
    
    def serialize_samples(self, writer, data_file, label_file):
        frames, labels, seq_num, num_samples, names = self._get_samples(data_file, label_file)
        
        # store data
        writer.create_dataset('visual', data=frames)
        writer.create_dataset('labels', data=labels)
        
        # Save meta-data
        writer.attrs['data_file'] = str(data_file)
        writer.attrs['label_file'] = str(label_file)
        writer.attrs['seq_num'] = seq_num
        writer.attrs['num_samples'] = num_samples 
        writer.attrs['label_names'] = names[1:]
    