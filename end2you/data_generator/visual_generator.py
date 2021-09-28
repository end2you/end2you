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
                 detector:FaceExtractor = None,
                 fps:int = 30, 
                 *args, **kwargs):
        """ Initialize object class.
        
        Args:
          labelfile_reader (FileReader): Class to read label files.
          detector (FaceExtractor): Instance of face detector (default FaceExtractor()).
          fps (int): The frames per second to use for the visual modality.
        """
        if not detector:
            detector = FaceExtractor()
        
        self.labelfile_reader = labelfile_reader
        self.fps = fps
        self.detector = detector
        super().__init__(*args, **kwargs)
    
    def _get_samples(self, data_file, label_file):
        """ Read samples from file.
        
        Args:
          data_file (str): File to read the data from.
          label_file (str): File to read the labels from.
        
        Returns:
          frames (np.array): Frames of each file.
          labels (np.array): Labels for each frame.
          seq_num (int): Number of samples in file.
          num_samples (int): Number of samples per frame.
          attrs_name (str): Label names.
        """

        file_data, attrs_name = self.labelfile_reader.read_file(label_file)
        file_data = np.array(file_data).astype(np.float32)
        timestamps = file_data[:,0]
        labels = file_data[:-1,1:]
        
        clip = VideoFileClip(str(data_file))
        
        if not self.fps:
            self.fps = int(clip.fps)
        
        seq_num = labels.shape[0] - 1
        num_samples = int(self.fps * (timestamps[1] - timestamps[0]))
        
        frames = []        
        for i in range(len(timestamps) - 1):
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            
            data_frame = np.array(list(clip.subclip(start_time, end_time).iter_frames()))
            data_frame = data_frame[:num_samples]
            
            if self.detector:
                try:
                    data_frame = self.detector.extract_and_resize_face(data_frame)
                except:
                    data_frame = np.zeros(
                        (1, self.detector.resize[0], self.detector.resize[0], 3), 
                        dtype=np.float32)
            
            if data_frame.shape[0] < num_samples:
                data_frame = np.pad(data_frame, (
                        (0,num_samples - data_frame.shape[0] % num_samples),
                        (0,0),(0,0),(0,0)), 'reflect')
            
            data_frame = data_frame.transpose(0, 3, 1, 2)
            
            frames.append(data_frame.astype(np.float32))
        
        frames = np.array(frames).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        
        return frames, labels, seq_num, num_samples, attrs_name
    
    def serialize_samples(self, writer, data_file, label_file):
        """ Write data to `hdf5` file.
        
        Args:
          writer (h5py.File): Open file to write data.
          data_file (str): Data file name.
          label_file (str): Label file name.
        """
        
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
    