import logging
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")

from end2you.data_provider import get_dataloader
from end2you.data_provider.hdf5 import BaseProvider
from end2you.base import BasePhase
from end2you.base_process import BaseProcess
from .metric_provider import MetricProvider
from tqdm import tqdm
from pathlib import Path


class Evaluator(BasePhase):
    
    def __init__(self,
                 metric:MetricProvider,
                 data_provider:BaseProvider,
                 model:nn.Module,
                 model_path:str,
                 cuda:bool,
                 root_dir = './',
                 take_last_frame:bool = True):
        """ Initialize object class to perform evaluation.
        
        Args:
            metric (MetricProvider): Metric to use for evaluation.
            data_provider (BaseProvider): Data provider.
            root_dir (str): Root directory.
            model (torch.nn.Module): Model to use for evaluation.
            model_path (str): Path to restore model.
            cuda (bool): Use GPU.
        """
        
        super().__init__(model, model_path)
        self.eval_fn = metric.eval_fn
        self.metric_name = metric.metric_name
        self.data_provider = data_provider
        self.model = model
        self.cuda = cuda
        self.root_dir = Path(root_dir)
        self.take_last_frame = take_last_frame
        BaseProcess.set_logger(str(self.root_dir / 'evaluation.log'))
    
    def start_evaluation(self):
        """ Perform one epoch of training or evaluation.
            Depends on the argument `is_training`.
        """
        
        logging.info("Starting Evaluation!")
        
        provider = self.data_provider
        
        # Load model
        self.load_checkpoint()
        
        # Put model for evaluation
        self.model.eval()
        
        summary_scores = []
        num_outs = self.model.num_outs
        file_preds = {str(x.file_path.name):[] for x in provider.dataset.data_files}
        label_names = provider.dataset._get_label_names()
        batch_preds = {str(x):[] for x in provider.dataset.label_names}
        batch_labels = {str(x):[] for x in provider.dataset.label_names}
        batch_masks = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(provider)) as bar:
            bar.set_description('Evaluating model')
            for n_iter, (model_input, labels, masked_samples, file_names) in enumerate(provider):
                
                # move to GPU if available
                if self.cuda:
                    model_input = [x.cuda() for x in model_input] if isinstance(model_input, list) else model_input.cuda()
                    labels = labels.cuda()
                
                predictions = self.model(model_input)
                for f in file_names:
                    file_preds[str(Path(f).name)].extend(predictions.data.cpu().numpy())
                
                np_preds = predictions.data.cpu().numpy()
                np_labels = labels.data.cpu().numpy()
                batch_masks.extend(masked_samples)
                for o, name in enumerate(label_names):
                    sl = o
                    el = o + 1 if len(label_names) > 1 else o + num_outs
                    batch_preds[name].extend(np_preds[...,sl:el])
                    batch_labels[name].extend(np_labels[...,sl:el])
                
                bar.update()
        
        scores = {}
        for i, name in enumerate(label_names):
            scores[name] = self.eval_fn(batch_preds[name], batch_labels[name], batch_masks, self.take_last_frame)
        epoch_summaries = [scores]
        
        # Reseting parameters of the data provider
        provider.dataset.reset()
        
        # compute mean of all metrics in summary
        mean_scores = {
            label_name: np.mean([
                batch_sum[label_name] for batch_sum in epoch_summaries
            ]) 
            for label_name in scores.keys()
        }
        
        str_list_scores = [f'{label_name}: {mean_scores[label_name]:05.3f}' 
                           for label_name in label_names]
        str_scores = ' - '.join(str_list_scores) 
        label_scores_mean = np.mean([
            mean_scores[label_name]  for label_name in label_names])
        
        logging.info(f'* Evaluation results (wrt {self.metric_name}): {str_scores}\n')
        
        if len(label_names) > 1:
            file_preds = {k:np.vstack(v).tolist() for k,v in file_preds.items()}
        else:
            file_preds = {k:np.argmax(v[0][-1]).tolist() for k,v in file_preds.items()}
        
        file_preds.update({'label_names': provider.dataset.label_names.tolist()})
        self._save_dict_to_json(file_preds, str(self.root_dir / 'predictions.json'))
        
        return label_scores_mean, file_preds