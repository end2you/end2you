from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    
    @abstractmethod
    def create_model(*args, **kwargs):
        pass
    
    