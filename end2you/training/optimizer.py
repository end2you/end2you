import torch.optim as optim


def get_optimizer(optimizer:str):
    """ Factory method to provide the optimizer. 
    
    Args:
      optimizer (str): The optimizer to use.
    """
    
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
