import argparse
import numpy as np
import math
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from end2you.add_parsers import add_parsers
from end2you.utils import Params
from end2you.training_process import TrainingProcess
from end2you.generation_process import GenerationProcess
from end2you.evaluation_process import EvaluationProcess

parser = add_parsers()


class End2You:
    
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
    
    def start_process(self):
        if 'train' in self.kwargs['process'].lower():
            train_params = self._get_train_params()
            training = TrainingProcess(train_params)
            training.start()
        elif 'test' in self.kwargs['process'].lower():
            eval_params = self._get_eval_params()
            evaluation = EvaluationProcess(eval_params)
            evaluation.start()
        elif 'generate' in self.kwargs['process'].lower():
            gen_params = self._get_gen_params()
            generation = GenerationProcess(gen_params)
            generation.start()
        else:
            raise ValueError('''Should indicate which operation to perform.'''
                ''' Valid one of [generate, train, test].''')
    
    def _get_train_params(self):
        cuda = True if 'true' == self.kwargs['cuda'].lower() else False
        pretrained = True if 'true' == self.kwargs['pretrained'].lower() else False
        take_last_frame = True if 'true' == self.kwargs['take_last_frame'].lower() else False
        
        train_params = Params(dict_params={
            'train':Params(dict_params={'loss':self.kwargs['loss'],
                                        'dataset_path':self.kwargs['train_dataset_path'],
                                        'optimizer':self.kwargs['optimizer'],
                                        'learning_rate':self.kwargs['learning_rate'],
                                        'num_epochs':self.kwargs['num_epochs'],
                                        'num_workers':self.kwargs['train_num_workers'],
                                        'cuda': cuda,
                                        'modality':self.kwargs['modality'],
                                        'batch_size':self.kwargs['batch_size'],
                                        'is_training':True,
                                        'save_summary_steps':self.kwargs['save_summary_steps'],
                                        'seq_length': self.kwargs['seq_length']
                                       }),
            'valid':Params(dict_params={'metric':self.kwargs['metric'],
                                        'dataset_path':self.kwargs['valid_dataset_path'],
                                        'num_workers':self.kwargs['valid_num_workers'],
                                        'cuda':cuda,
                                        'modality':self.kwargs['modality'],
                                        'batch_size':1,
                                        'save_summary_steps':1,
                                        'is_training':False,
                                        'seq_length': self.kwargs['seq_length']
                                      }),
            'model':Params(dict_params={'model_name':self.kwargs['model_name'],
                                        'pretrained':pretrained, 
                                        'num_outs':self.kwargs['num_outputs']}),
            'root_dir': self.kwargs['root_dir'],
            'log_file': self.kwargs['log_file'],
            'ckpt_path': self.kwargs['ckpt_path'],
            'take_last_frame':take_last_frame
        })
        return train_params
    
    def _get_eval_params(self):
        cuda = True if 'true' == self.kwargs['cuda'].lower() else False
        take_last_frame = True if 'true' == self.kwargs['take_last_frame'].lower() else False
        
        eval_params = Params(dict_params={
            'prediction_file': self.kwargs['prediction_file'],
            'seq_length': self.kwargs['seq_length'],
            'dataset_path': self.kwargs['dataset_path'],
            'model_path': self.kwargs['model_path'],
            'metric': self.kwargs['metric'],
            'num_workers': self.kwargs['num_workers'],
            'cuda': cuda,
            'num_gpus': self.kwargs['num_gpus'],
            'modality': self.kwargs['modality'],
            'log_file': self.kwargs['log_file'],
            'root_dir': self.kwargs['root_dir'],
            'take_last_frame':take_last_frame,
            'model':Params(dict_params={'model_name':self.kwargs['model_name'],
                                        'num_outs':self.kwargs['num_outputs']})
        })
        
        return eval_params
    
    def _get_gen_params(self):
        generator_params = Params(dict_params={
            'save_data_folder': self.kwargs['save_data_folder'],
            'modality': self.kwargs['modality'],
            'input_file': self.kwargs['input_file'],
            'exclude_cols': self.kwargs['exclude_cols'],
            'delimiter': self.kwargs['delimiter'],
            'fieldnames': self.kwargs['fieldnames'],
            'log_file': self.kwargs['log_file'],
            'root_dir': self.kwargs['root_dir']
        })
        
        return generator_params


def main(flags=None):
    
    if flags == None:
        flags=sys.argv[1:]
    flags = vars(parser.parse_args(flags))
    
    e2u = End2You(**flags)
    e2u.start_process()


if __name__ == '__main__':
    main()
