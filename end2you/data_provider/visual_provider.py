import torch

from .provider import BaseProvider
from torchvision import transforms


class VisualProvider(BaseProvider):
    '''VisualProvider dataset.'''
    
    def __init__(self,*args, **kwargs):
        self.modality = 'visual'
        super().__init__(*args, **kwargs)
    
    def _init_augment(self):
        return transforms.Compose([
                transforms.RandomAffine(180),
                transforms.RandomResizedCrop(96)
            ])
    
    def process_input(self, data, labels):
        data = data[:,0,...]/255.
        if self.augment:
            data = self.data_transform(torch.Tensor(data))
        
        return data, labels
