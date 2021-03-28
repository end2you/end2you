import torch
import torch.optim as optim
import logging
import json 

from torch import nn
from pathlib import Path


class BasePhase:
    """ Base class for train/eval models."""
    
    def __init__(self,
                 model:nn.Module,
                 ckpt_path:str = None,
                 optimizer:optim = None):
        """ Initialize object class.
        
        Args:
            model (torch.nn.Module): Model to train/evaluate.
            ckpt_path (str): Path to the pretrain model.
            optimizer (torch.optim): Optimizer to use for training.
        """
        
        self.optimizer = optimizer
        self.model = model
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
    
    def load_checkpoint(self):
        """ Loads model parameters (state_dict) from file_path. 
            If optimizer is provided, loads state_dict of
            optimizer assuming it is present in checkpoint.
        
        Args:
            checkpoint (str): Filename which needs to be loaded
            model (torch.nn.Module): Model for which the parameters are loaded
            optimizer (torch.optim): Optional: resume optimizer from checkpoint
        """
        
        logging.info("Restoring model from [{}]".format(str(self.ckpt_path)))
        
        if not Path(self.ckpt_path).exists():
            raise Exception("File doesn't exist [{}]".format(str(self.ckpt_path)))
        checkpoint = torch.load(str(self.ckpt_path))
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optim_dict'])
        
        return checkpoint
    
    def _save_dict_to_json(self, 
                           dictionary:dict, 
                           json_path:str):
        """ Saves dict of floats in json file
        
        Args:
            dictionary (dict): of float-castable values (np.float, int, float, etc.)
            json_path (string): path to json file
        """
        
        with open(json_path, 'w') as f:
            json.dump(dictionary, f, indent=4)
    