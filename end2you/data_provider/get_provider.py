import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from .audio_provider import AudioProvider
from .visual_provider import VisualProvider
from .multifile_audiovisual_provider import MultiFile_AVProvider
from .singlefile_audiovisual_provider import SingleFile_AVProvider


def get_provider(modality):
    return {
        'audio': AudioProvider,
        'visual': VisualProvider,
        'audiovisual': SingleFile_AVProvider
    }[modality]


def pad_collate(batch):
    data, labels, data_file = zip(*batch)
    
    number_of_modalities = len(data[0]) if isinstance(data[0], list) else 1
    if number_of_modalities > 1:
        data = data.unsqueeze(1)
    
    modality_tensors = []
    for i in range(number_of_modalities):
        modality_i = [torch.Tensor(x) for x in data]
        padded_modality = pad_sequence(modality_i, batch_first=True)
        modality_tensors.append(padded_modality)
    
    num_seqs_per_sample = [len(x) for x in labels]
    labels = [torch.Tensor(x) for x in labels]
    labels = pad_sequence(labels, batch_first=True)
    
    if number_of_modalities == 1:
        modality_tensors = modality_tensors[0]
    
    return modality_tensors, labels, num_seqs_per_sample, data_file


def get_dataloader(params, **kwargs):
    """
    Gets the DataLoader object for each type in split_dirs keys.
    Args:
        params (Params)  : contains `num_workers` and `cuda` parameters
    Returns:
        dataloaders (dict): contains the DataLoader object for each type in `split_dirs` keys
    """
    Provider = get_provider(params.modality)
    return  DataLoader(Provider(params.dataset_path, seq_length=params.seq_length),
                       batch_size=params.batch_size,
                       shuffle=params.is_training,
                       num_workers=params.num_workers,
                       pin_memory=params.cuda,
                       collate_fn=pad_collate)
