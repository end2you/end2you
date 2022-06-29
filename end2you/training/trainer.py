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
from end2you.base_process import BaseProcess


class Trainer(BasePhase):
    
    def __init__(self,
                 loss:Losses,
                 evaluator:MetricProvider,
                 data_providers:dict,
                 summary_writers:dict,
                 root_dir:str,
                 model:nn.Module,
                 ckpt_path:str,
                 optimizer:torch.optim,
                 params:Params):
        """ Initialize trainer object class.
        
        Args:
          loss (Losses): The loss function to use.
          evaluator (MetricProvider): The evaluation function to use.
          data_providers (dict): The training/evaluation data providers.
          summary_writers (dict): The training/evaluation summary writers.
          root_dir (str): Directory path to save output files (e.g. models)
          model (nn.Module): Instance of a model to train.
          ckpt_path (str): Path to pre-train model.
          optimizer (torch.optim): Instance of an optimizer to use.
          params (Params): Rest of training parameters.
        """
        
        params.valid.dict['save_summary_steps'] = 1
        self.params = params
        
        self.root_dir = Path(params.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.loss_fn = loss.loss_fn
        self.loss_name = loss.loss_name.upper()
        
        self.eval_fn = evaluator.eval_fn
        self.metric = evaluator.metric_name
        
        self.provider = data_providers
        self.summary_writer = summary_writers
        BaseProcess.set_logger('training.log')
        super().__init__(model, ckpt_path, optimizer)
    
    def start_training(self):
        """ Method that performs the training of the model.
        """
        
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
            self._epoch_process(is_training=True)
            
            # Evaluate for one epoch on validation set
            with torch.no_grad():
                val_score = self._epoch_process(is_training=False)
            
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
                best_json_path = str(self.root_dir / "best_valid_scores.json")
                self._save_dict_to_json({self.metric:val_score}, str(best_json_path))
            
            # Save latest val metrics in a json file in the model directory
            last_json_path = str(self.root_dir / "last_valid_scores.json")
            self._save_dict_to_json({self.metric:val_score}, str(last_json_path))
    
    def _write_bestscore(self, best_score:str):
        """ Method to write best current score to a file.
        
        Args:
          best_score (str): Best score to save to `best_score.txt` file.
        """
        
        f = open(str(self.root_dir / "best_score.txt"),"w+")
        f.write(f"Best score: {best_score}")
        f.close()
    
    def save_checkpoint(self, state:dict, is_best:bool, checkpoint:str):
        """ Saves model and training parameters at checkpoint + 'last.pth.tar'. 
            If is_best==True, also saves checkpoint + 'best.pth.tar'
        
        Args:
            state (dict): contains model's state_dict, and some other info of the model.
            is_best (bool): True if it is the best model seen till now.
            checkpoint (str): Folder to save model.
        """
        
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(exist_ok=True)
        
        filepath = checkpoint / 'last.pth.tar'
        
        torch.save(state, str(filepath))
        if is_best:
            shutil.copyfile(filepath, str(checkpoint / 'best.pth.tar'))
    
    def _epoch_process(self, is_training:bool):
        """ Perform one epoch of training or evaluation.
            Depends on the argument `is_training`.
        """
        
        params = self.params.train if is_training else self.params.valid
        process = 'train' if is_training else 'valid'
        
        writer = self.summary_writer[process]
        provider = self.provider[process]
        
        label_names = provider.dataset._get_label_names()
        
        num_outs = self.model.num_outs if not isinstance(
            self.model, nn.DataParallel) else self.model.module.num_outs
        
        self.model.train(is_training)
        
        # summary for current training loop and a running average object for loss
        mean_loss = 0.0
        
        # Store all labels/predictions 
        batch_labels = {str(x):[] for x in provider.dataset.label_names}
        batch_preds = {str(x):[] for x in provider.dataset.label_names}
        batch_masks = []
        
        bar_string = 'Training' if process == 'train' else 'Validating'
        
        # Use tqdm for progress bar
        with tqdm(total=len(provider)) as bar:
            bar.set_description(f'{bar_string} model')
            for n_iter, (model_input, labels, masked_samples, _) in enumerate(provider):
                
                input_dtype = model_input[0].dtype if isinstance(model_input, list) \
                                                   else model_input.dtype 
                
                if is_training:
                    self.optimizer.zero_grad()
                
                # move to GPU if available
                if params.cuda:
                    model_input = [
                        x.cuda() for x in model_input] if isinstance(model_input, list) \
                                                       else model_input.cuda()
                    labels = labels.cuda()
                
                predictions = self.model(model_input)
                
                total_loss = nn.Parameter(
                    torch.tensor(0.0, dtype=input_dtype), 
                    requires_grad=is_training)
                for o, name in enumerate(label_names):
                    sl = o
                    el = o + 1 if len(label_names) > 1 else o + num_outs
                    
                    label_loss = self.loss_fn(
                        predictions[...,sl:el], labels[...,sl:el], masked_samples, self.params.take_last_frame)
                    
                    total_loss = total_loss + label_loss
                    
                    # Write label summary
                    if n_iter % params.save_summary_steps == 0:
                        writer.add_scalar(f'{self.loss_name}_loss_{name}/', label_loss)
                
                total_loss /= num_outs
                mean_loss += total_loss
                
                if is_training:
                    total_loss.backward()
                    self.optimizer.step()
                
                # Evaluate summaries once in a while
                if n_iter % params.save_summary_steps == 0:
                    
                    batch_loss = total_loss.item()
                    np_preds = predictions.data.cpu().numpy()
                    np_labels = labels.data.cpu().numpy()
                    
                    batch_masks.extend(masked_samples)
                    for o, name in enumerate(label_names):
                        sl = o
                        el = o + 1 if len(label_names) > 1 else o + num_outs
                        batch_preds[name].extend(np_preds[...,sl:el])
                        batch_labels[name].extend(np_labels[...,sl:el])
                
                bar.set_postfix({self.loss_name+' loss':'{:05.3f}'.format(total_loss.item())}) 
                bar.update()
        
        scores = {}
        for i, name in enumerate(label_names):
            scores[name] = self.eval_fn(batch_preds[name], batch_labels[name], batch_masks, self.params.take_last_frame)
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
