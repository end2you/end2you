import numpy as np
import torch

from facenet_pytorch import MTCNN
from PIL import Image


class FaceExtractor:
    
    def __init__(self,
                 resize:tuple = (96, 96)):
        """ Initialize face extractor object.
        
        Args:
          resize (tuple): Size to resize frames (default `(96,96)`).
        """
        
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(keep_all=True, device=device)
        self.resize = resize
    
    def resize_frames(self, frames:list):
        ''' Resizes frames.
        
        Args:
            frames (list): List of frames.
        '''
        
        resized_frames = [] 
        for frame in frames:
            resized_img = Image.fromarray(frame).resize(self.resize)
            resized_frames.append(np.array(resized_img))
        
        return np.array(resized_frames)
    
    def extract_face(self, frames:np.array):
        ''' Detects and extract the most visible face from a frame.
        
        Args:
            frames (np.array) (N x H x W x 3): N frames
        
        Returns:
            The cropped face of the subject.
        '''
        
        # detect faces in the image
        results = self.detector.detect(frames)
        
        cropped_frames = []
        for i, frame in enumerate(frames):
            
            if max(results[1][i]):
                idx = np.argmax(results[1][i])
                
                # extract the bounding box from the first face
                x1, y1, x2, y2 = results[0][i][idx].astype(int)
                x1 = max(x1,0)
                y1 = max(y1,0)
                x2 = min(x2,frame.shape[0])
                y2 = min(y2,frame.shape[1])
                
                # extract the face
                frame = frame[y1:y2, x1:x2]
            cropped_frames.append(frame)
        
        return cropped_frames
    
    def extract_and_resize_face(self, frames:list):
        frames = self.extract_face(frames)
        frames = self.resize_frames(frames)
        return frames
    