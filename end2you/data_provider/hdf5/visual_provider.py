import torch

from .base_provider import BaseProvider
from torchvision import transforms


class VisualProvider(BaseProvider):
    """Provides the data for the visual modality."""
    
    def __init__(self,*args, **kwargs):
        self.modality = 'visual'
        super().__init__(*args, **kwargs)
    
    def _init_augment(self):
        """ Augmentation method for visual data. 
            2 augmentaitons are provided:
              RandomAffine(180)
              RandomResizedCrop(96)
        """
        return transforms.Compose([
                transforms.RandomAffine(180),
                transforms.RandomResizedCrop(96)
            ])
    
    def process_input(self, data, labels):
        """ Pre-process input frames to be in the range [0, 1]. 
            
        Args:
          data (np.array): Input frames.
          labels (np.array): Labels.
        """
        data = data[:,0,...]/255.
        if self.augment:
            data = self.data_transform(torch.Tensor(data))
        
        return data, labels
