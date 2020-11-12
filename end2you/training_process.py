import end2you.models as models
import logging
import torch
import end2you.training.optimizer as optim

from end2you.training import Losses, Trainer
from end2you.data_provider import get_dataloader
from end2you.utils import Params
from torch.utils.tensorboard import SummaryWriter
from end2you.evaluation import MetricProvider
from pathlib import Path


class TrainingProcess:
    
    def __init__(self, 
                 params:Params):
        
        loss_cls = Losses(params.train.loss)
        eval_fn = MetricProvider(params.valid.metric)
        processes = ['train', 'valid']
        data_providers = {
            process: get_dataloader(getattr(params, process)) for process in processes
        }
        
        params.model.dict['input_size'] = data_providers['train'].dataset._get_frame_num_samples()
        
        # Model
        model = models.get_model(params.train.modality, **params.model.dict)
        num_model_params = [
            pmt.numel() for pmt in model.parameters() if pmt.requires_grad is True]
        
        optimizer = optim.get_optimizer(params.train.optimizer)
        optimizer = optimizer(model.parameters(), lr=params.train.learning_rate)
        
        # Use GPU
        if params.train.cuda:
            gpus = [str(x) for x in range(params.num_gpus)]
            device = torch.device("cuda:{}".format(','.join(gpus)))
            torch.cuda.set_device(device)
            
            if torch.cuda.device_count() > 1:
                logging.info('Using', torch.cuda.device_count(), 'GPUs!')
                model = nn.DataParallel(model)
            
            model.to(device)
        
        tb_path = Path(params.root_dir) / 'tensorboard' 
        summary_writers = {
            process: SummaryWriter(str(tb_path / getattr(params, process).summarywriter_file))
                for process in processes
        }
        
        # Initialize logs
        self.set_logger(params.log_file)
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
        self.trainer.start_training()
    