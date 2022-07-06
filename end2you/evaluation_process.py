import torch
import torch.nn as nn
import logging
import end2you.models as models

from end2you.data_provider import get_dataloader
from end2you.utils import Params
from end2you.evaluation import MetricProvider, Evaluator
from end2you.base_process import BaseProcess
from pathlib import Path


class EvaluationProcess(BaseProcess):
    """ Process evaluation class."""
    
    def __init__(self, 
                 params:Params,
                 *args, **kwargs):
        
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
        log_file = Path(params.root_dir) / params.log_file
        self.set_logger(str(log_file))
        logging.info('Starting Evaluation Process')
        logging.info('Number of parameters: {}'.format(sum(num_model_params)))
        logging.info(model)
        
        self.evaluator = Evaluator(metric=eval_fn,
                                   data_provider=data_provider,
                                   model=model,
                                   model_path=params.model_path,
                                   cuda=params.cuda,
                                   root_dir=params.root_dir,
                                   take_last_frame=params.take_last_frame)
    
    def start(self):
        return self.evaluator.start_evaluation()
