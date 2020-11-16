import torch.optim as optim


def get_optimizer(optimizer:str):
    ''' Provides the optimizer.''' 
    return {
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sparseadam':optim.SparseAdam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD,
        'adadelta':optim.Adadelta
    }[optimizer]
