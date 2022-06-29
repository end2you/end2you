import end2you.models as models
import logging
import torch
import torch.nn as nn
import end2you.training.optimizer as optim

from end2you.training import Losses, Trainer
from end2you.data_provider import get_dataloader
from end2you.utils import Params
from torch.utils.tensorboard import SummaryWriter
from end2you.evaluation import MetricProvider
from pathlib import Path
from end2you.base_process import BaseProcess


class TrainingProcess(BaseProcess):
    """ Training process class. """
    
    def __init__(self, 
                 params:Params,
                 *args, **kwargs):
        
        # Define Loss function
        loss_cls = Losses(params.train.loss)
        
        # Define Metric function
        eval_fn = MetricProvider(params.valid.metric)
        
        # Get training/validation data provider
        processes = ['train', 'valid']
        data_providers = {
            process: get_dataloader(getattr(params, process)) for process in processes
        }
        
        params.model.dict['input_size'] = data_providers['train'].dataset._get_frame_num_samples()
        params = self._update_params(params)
        
        # Model
        model = models.get_model(params.train.modality, **params.model.dict)
        num_model_params = [
            pmt.numel() for pmt in model.parameters() if pmt.requires_grad is True]
        
        # Get optimizer to train model
        optimizer = optim.get_optimizer(params.train.optimizer)
        optimizer = optimizer(model.parameters(), lr=params.train.learning_rate)
        
        # Set device
        device = torch.device("cuda:0" if params.train.cuda else "cpu")
        
        # Use Multiple GPUs
        if params.train.cuda:
            if torch.cuda.device_count() > 1:
                logging.info('Using', torch.cuda.device_count(), 'GPUs!')
                model = nn.DataParallel(model)
        
        model.to(device)
        
        # Get training/validation Summary writers for tensorboard visualization
        tb_path = Path(params.root_dir) / 'summarywriters' 
        summary_writers = {
            process: SummaryWriter(str(tb_path / process))
                for process in processes
        }
        
        # Initialize logs
        log_file = Path(params.root_dir) / params.log_file
        self.set_logger(str(log_file))
        logging.info('Starting Training Process')
        logging.info('Number of parameters: {}'.format(sum(num_model_params)))
        logging.info(model)
        
        self.trainer = Trainer(loss=loss_cls, 
                               evaluator=eval_fn,
                               data_providers=data_providers,
                               summary_writers=summary_writers,
                               root_dir=params.root_dir,
                               model=model,
                               ckpt_path=params.ckpt_path,
                               optimizer=optimizer,
                               params=params)
    
    def _update_params(self, params:Params):
        params.train.dict['is_training'] = True
        params.train.dict['num_workers'] = 0
        
        params.valid.dict['save_summary_steps'] = 1
        params.valid.dict['is_training'] = False
        params.valid.dict['cuda'] = params.train.cuda
        params.valid.dict['num_workers'] = 0
        params.valid.dict['augment'] = False
        
        return params
    
    def start(self):
        self.trainer.start_training()
