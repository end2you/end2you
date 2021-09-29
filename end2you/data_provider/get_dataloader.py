import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from .get_provider import *


def pad_collate(batch):
    """ Pad batch tensors to have equal length.
    
    Args:
        batch (list): Data to pad.
    
    Returns:
        modality_tensors (torch.Tensor): Batched data tensors. 
        labels (torch.Tensor): Batched label tensors.
        num_seqs_per_sample (list): Number of sequences of each batch tensor.
        data_file (str): File name.
    """
    
    data, labels, data_file = zip(*batch)
    
    number_of_modalities = len(data[0]) if isinstance(data[0], list) else 1
    if number_of_modalities == 1:
        data = [[x] for x in data]

    modality_tensors = []
    for i in range(number_of_modalities):
        modality_i = [torch.Tensor(x[i]) for x in data]
        padded_modality = pad_sequence(modality_i, batch_first=True)
        modality_tensors.append(padded_modality)
    
    num_seqs_per_sample = [len(x) for x in labels]
    labels = [torch.Tensor(x) for x in labels]
    labels = pad_sequence(labels, batch_first=True)
    
    if number_of_modalities == 1:
        modality_tensors = modality_tensors[0]
    
    return modality_tensors, labels, num_seqs_per_sample, data_file


def get_dataloader(params, format_name='hdf5', **kwargs):
    """ Gets the DataLoader object for each type in split_dirs keys.
    
    Args:
      params (Params) : Parameters needed to load data.
        `modality` (str): Modality to provide data from.
        `dataset_path` (str): Path to `hdf5` data files.
        `seq_length` (int): Number of consecuvite frames to load.
        `batch_size` (int): Batch size.
        `cuda` (int): Whether to use cuda
        `num_workers` (int): Number of workers to use.
        `is_training` (bool): Whether to provide data for training/evaluation.
    
    Returns:
      dataloaders (dict): contains the DataLoader object for each type 
                          in `split_dirs` keys.
    """
    
    Provider = get_proper_provider(format_name)(params.modality)
    
    return  DataLoader(Provider(params.dataset_path,
                                seq_length=params.seq_length),
                       batch_size=params.batch_size,
                       shuffle=params.is_training,
                       num_workers=params.num_workers,
                       pin_memory=params.cuda,
                       collate_fn=pad_collate)
