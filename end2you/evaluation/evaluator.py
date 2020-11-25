import logging
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")

from end2you.data_provider import get_dataloader, BaseProvider
from end2you.base import BasePhase
from end2you.base_process import BaseProcess
from .metric_provider import MetricProvider
from tqdm import tqdm
from pathlib import Path


class Evaluator(BasePhase):
    '''Evaluation class.'''
    
    def __init__(self,
                 metric:MetricProvider,
                 data_provider:BaseProvider,
                 model:nn.Module,
                 model_path:str,
                 cuda:bool,
                 root_dir = './'):
        '''
        Args:
            metric (MetricProvider): Metric to use for evaluation.
            data_provider (BaseProvider): Data provider.
            root_dir (str): Root directory.
            model (torch.nn.Module): Model to use for evaluation.
            model_path (str): Path to restore model.
            cuda (bool): Use GPU.
        '''
        super().__init__(model, model_path)
        self.eval_fn = metric.eval_fn
        self.metric_name = metric.metric_name
        self.data_provider = data_provider
        self.model = model
        self.cuda = cuda
        self.root_dir = Path(root_dir)
        BaseProcess.set_logger(str(self.root_dir / 'evaluation.log'))
    
    def start_evaluation(self):
        '''
           Perform one epoch of training or evaluation.
           Depends on the argument `is_training`.
        '''
        logging.info("Starting Evaluation!")
        
        provider = self.data_provider
        
        # Load model
        self.load_checkpoint()
        
        # Put model for evaluation
        self.model.eval()
        
        summary_scores = []
        file_preds = {str(x.file_path.name):[] for x in provider.dataset.data_files}
        # Use tqdm for progress bar
        with tqdm(total=len(provider)) as bar:
            bar.set_description('Evaluating model')
            for n_iter, (model_input, labels, masked_samples, file_names) in enumerate(provider):
                
                # move to GPU if available
                if self.cuda:
                    model_input = [x.cuda() for x in model_input] if isinstance(model_input, list) else model_input.cuda()
                    labels = labels.cuda()
                
                output = self.model(model_input)
                for f in file_names:
                    file_preds[str(f.name)].extend(output.data.cpu().numpy())
                
                # compute all metrics on this batch
                scores = {}
                for i, name in enumerate(provider.dataset.label_names):
                    scores[name] = self.eval_fn(output[...,i], labels[...,i], masked_samples)
                
                summary_scores.append(scores)
                bar.update()
        
        # Reseting parameters of the data provider
        provider.dataset.reset()
        
        # compute mean of all metrics in summary
        mean_scores = {label_name:np.mean([batch_sum[label_name] for batch_sum in summary_scores]) 
                             for label_name in scores.keys()}
        
        str_list_scores = [f'{label_name}: {mean_scores[label_name]:05.3f}' 
                           for label_name in provider.dataset.label_names]
        str_scores = ' - '.join(str_list_scores)
        logging.info(f'* Evaluation results (wrt {self.metric_name}): {str_scores}\n')
        
        file_preds = {k:np.vstack(v).tolist() for k,v in file_preds.items()}
        file_preds.update({'label_names': provider.dataset.label_names.tolist()})
        self._save_dict_to_json(file_preds, str(self.root_dir / 'predictions.json'))

        return mean_scores, file_preds
