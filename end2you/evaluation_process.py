import torch
import torch.nn as nn
import end2you.models as models
import logging

from end2you.data_provider import get_dataloader
from end2you.utils import Params
from end2you.evaluation import MetricProvider, Evaluator


class EvaluationProcess:
    
    def __init__(self, 
                 params:Params):
        
        eval_fn = MetricProvider(params.metric)
        params.dict.update({'batch_size':1, 'is_training':False})
        data_provider = get_dataloader(params)
        
        params.model.dict['input_size'] = data_provider.dataset.data_files[0]._get_num_samples()
        
        # Model
        model = models.get_model(params.modality, **params.model.dict)
        num_model_params = [
            pmt.numel() for pmt in model.parameters() if pmt.requires_grad is True]
        
        # Use GPU
        if params.cuda:
            gpus = [str(x) for x in range(params.num_gpus)]
            device = torch.device("cuda:{}".format(','.join(gpus)))
            torch.cuda.set_device(device)
            
            if torch.cuda.device_count() > 1:
                print('Using', torch.cuda.device_count(), 'GPUs!')
                model = nn.DataParallel(model)
            
            model.to(device)
        
        # Initialize logs
        self.set_logger(params.log_file)
        logging.info('Number of parameters: {}'.format(sum(num_model_params)))
        logging.info(model)
        
        self.evaluator = Evaluator(metric=eval_fn,
                                   data_provider=data_provider,
                                   model=model,
                                   model_path=params.model_path,
                                   cuda=params.cuda)
    
    def set_logger(self, log_file:str):
        '''Set the logger to log info in terminal and file `log_path`.
                
        Args:
            log_path: (string) path to save log file.
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
        logging.info('**** Starting logging ****')
    
    def start(self):
        return self.evaluator.start_evaluation()
    