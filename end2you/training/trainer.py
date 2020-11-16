import torch
import torch.nn.functional as F
import torch.optim as optim
import end2you.utils as utils
import copy
import os
import numpy as np
import logging
import shutil
import sys
sys.path.append("..")

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from .losses import Losses
from end2you.base import BasePhase
from end2you.utils import Params
from end2you.evaluation import MetricProvider


class Trainer(BasePhase):
    """Train class."""
    
    def __init__(self,
                 loss:Losses,
                 evaluator:MetricProvider,
                 data_providers:dict,
                 summary_writers:dict,
                 root_dir:str,
                 model:nn.Module,
                 ckpt_path:str,
                 optimizer:optim,
                 params:Params):
        
        self.params = params
        
        self.root_dir = Path(params.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.loss_fn = loss.loss_fn
        self.loss_name = loss.loss_name.upper()
        
        self.eval_fn = evaluator.eval_fn
        self.metric = evaluator.metric_name
        
        self.provider = data_providers
        self.summary_writer = summary_writers
        
        super().__init__(model, ckpt_path, optimizer)
    
    def start_training(self):
        logging.info("Starting training!")
        
        best_score = float('-inf')
        if self.ckpt_path:
            ckpt = self.load_checkpoint()
            best_score = ckpt['validation_score']
            logging.info(f'Model\'s score: {best_score}')
        
        save_ckpt_path = self.root_dir / 'model'
        for epoch in range(self.params.train.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, self.params.train.num_epochs))
            
            # compute number of batches in one epoch (one full pass over the training set)
            self._epoch_process(self.params.train.save_summary_steps, True)
            
            # Free cuda memory
            torch.cuda.empty_cache()
            
            # Evaluate for one epoch on validation set
            with torch.no_grad():
                val_score = self._epoch_process(self.params.valid.save_summary_steps, False)
            
            is_best = val_score >= best_score
            
            model_info = {
                'validation_score': val_score,
                'metric_name': f'{self.metric}',
                'loss_name': f'{self.loss_name}'
            }
            dict2save = {'epoch': epoch + 1,
                         'state_dict': self.model.state_dict(),
                         'optim_dict' : self.optimizer.state_dict()}
            dict2save.update(model_info)
            
            # Save weights
            self.save_checkpoint(dict2save,  
                                 is_best=is_best,
                                 checkpoint=save_ckpt_path)
            
            # If best model save it.
            if is_best:
                logging.info(f"- Found new best model with mean {self.metric}: {val_score:05.3f}")
                best_score = val_score
                
                # Write best.txt file
                self._write_bestscore(best_score)
                
                # Save best val metrics in a json file in the model directory
                best_json_path = str(self.root_dir / "best_metrics_evaluation.json")
                self._save_dict_to_json({self.metric:val_score}, str(best_json_path))
            
            # Save latest val metrics in a json file in the model directory
            last_json_path = str(self.root_dir / "last_metrics_evaluation.json")
            self._save_dict_to_json({self.metric:val_score}, str(last_json_path))
    
    def _write_bestscore(self, best_score):
        f = open(str(self.root_dir / "best_score.txt"),"w+")
        f.write("sss: {}".format(best_score))
        f.close()
    
    def save_checkpoint(self, state, is_best, checkpoint):
        """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
        checkpoint + 'best.pth.tar'
        
        Args:
            state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
            is_best: (bool) True if it is the best model seen till now
            checkpoint: (string) folder where parameters are to be saved
        """
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(exist_ok=True)
        
        filepath = checkpoint / 'last.pth.tar'
        
        torch.save(state, str(filepath))
        if is_best:
            shutil.copyfile(filepath, str(checkpoint / 'best.pth.tar'))
    
    def _epoch_process(self, process_params, is_training):
        """
        Perform one epoch of training or evaluation.
        Depends on the argument `is_training`.
        """
        params = self.params.train if is_training else self.params.valid
        process = 'train' if is_training else 'valid'
        
        writer = self.summary_writer[process]
        provider = self.provider[process]
        label_names = provider.dataset._get_label_names()
        self.model.train(is_training)
        
        # summary for current training loop and a running average object for loss
        epoch_summaries = []
        mean_loss = 0.0
        
        bar_string = 'Training' if process == 'train' else 'Validating'
        
        # Use tqdm for progress bar
        with tqdm(total=len(provider)) as bar:
            bar.set_description(f'{bar_string} model')
            for n_iter, (model_input, labels, masked_samples) in enumerate(provider):
                
                if is_training:
                    self.optimizer.zero_grad()
                
                # move to GPU if available
                if params.cuda:
                    model_input = [x.cuda() for x in model_input] if isinstance(model_input, list) else model_input.cuda()
                    labels = labels.cuda()
                
                output = self.model(model_input)
                
                total_loss = torch.tensor(0.0, requires_grad=is_training)
                for o in range(self.params.model.num_outs):
                    total_loss = total_loss + self.loss_fn(output[...,o], labels[...,o], masked_samples)
                total_loss /= self.params.model.num_outs
                mean_loss += total_loss
                
                if is_training:
                    total_loss.backward()
                    self.optimizer.step()
                
                # Evaluate summaries once in a while
                if n_iter % params.save_summary_steps == 0:
                    
                    batch_loss = total_loss.item()
                    
                    # compute all metrics on this batch
                    scores = {}
                    for i, name in enumerate(label_names):
                        scores[name] = self.eval_fn(output[...,i], labels[...,i], masked_samples)
                    
                    scores[f'{self.loss_name}_loss'] = batch_loss
                    epoch_summaries.append(scores)
                
                bar.set_postfix({self.loss_name+' loss':'{:05.3f}'.format(total_loss.item())}) 
                bar.update()
        
        # Reseting parameters of the data provider
        provider.dataset.reset()
        
        # compute mean of all metrics in summary
        mean_scores = {label_name:np.mean([batch_sum[label_name] for batch_sum in epoch_summaries]) 
                             for label_name in scores.keys()}
        mean_loss /= (n_iter + 1)
        
        str_list_scores = [f'{label_name}: {mean_scores[label_name]:05.3f}' 
                           for label_name in label_names]
        str_scores = ' - '.join(str_list_scores) 
        str_scores = str_scores + f' || {self.loss_name} loss: {mean_loss:05.3f}'
        logging.info(f'* {process} results (wrt {self.metric}): {str_scores}\n')
        
        for label_name in label_names:
            writer.add_scalar(
                f'{process}_evaluation/{label_name}', mean_scores[label_name])
        writer.add_scalar('loss/', mean_loss)
        
        label_scores_mean = np.mean([
            mean_scores[label_name]  for label_name in label_names])
        
        return label_scores_mean