import logging


class BaseProcess:
    '''Base process for geneation/trainining/evaluation processes.'''
    
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def set_logger(cls, log_file:str):
        '''Set the logger to log info in terminal and file `log_path`.
        
        Args:
            log_path(str): Path to save log file.
        '''
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)
            
            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)
        logging.info('**** Start logging ****')
    
    def start(self):
        raise NotImplementedError('Method not implemented!')
