import argparse
import numpy as np
import tensorflow as tf
import math

from end2you.models.audio_model import AudioModel
from end2you.models.video_model import VideoModel
from end2you.models.rnn_model import RNNModel
from end2you.models.base import fully_connected
from end2you.rw.file_reader import FileReader
from end2you.data_provider.unimodal_provider import UnimodalProvider 
from end2you.training import *
from end2you.evaluation import Eval
from end2you.tfrecord_generator.generate_unimodal import UnimodalGenerator
from end2you.tfrecord_generator.generate_multimodal import MultimodalGenerator
from end2you.parser import *
from pathlib import Path

slim = tf.contrib.slim


parser = argparse.ArgumentParser(description='End2You flags.')

parser.add_argument('--tfrecords_folder', type=Path,
                    help='The tfrecords directory.')
parser.add_argument('--batch_size', type=int, default=2,
                    help='The batch size to use.')
parser.add_argument('--input_type', type=str,
                    help='Which model is going to be used: audio, video, or both.',
                    choices=['audio', 'video', 'both'])
parser.add_argument('--hidden_units', type=int, default=128,
                    help='The number of hidden units in the RNN model.')
parser.add_argument('--seq_length', type=int, default=150,
                    help='The sequence length to introduce to the RNN.'
                         'if 0 seq_length will be introduced' 
                         'by the audio model.')
parser.add_argument('--task', type=str, default='classification',
                    help='The number of epochs to run training (default 10).')
parser.add_argument('--num_classes', type=int, default=3,
                    help='If the task is classification the number of classes to consider.')

subparsers = parser.add_subparsers(help='Depending on --option value', dest='which')

training_subparser = subparsers.add_parser('train', help='Training argument options')
training_subparser = add_train_args(training_subparser)

evaluation_subparser = subparsers.add_parser('evaluate', help='Evaluation argument options')
evaluation_subparser = add_eval_args(evaluation_subparser)

generation_subparser = subparsers.add_parser('generate', help='Generation arguments')
generation_subparser = add_gen_args(generation_subparser)


class End2You:
    
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
    
    def _reshape_to_rnn(self, frames):
        batch_size, num_features = frames.get_shape().as_list()
        seq_length = self.kwargs['seq_length'] 
        if seq_length == 0:
            seq_length = -1
        frames = tf.reshape(frames, [self.kwargs['batch_size'], seq_length, num_features])
        
        return frames
    
    def _reshape_to_conv(self, frames):
        frame_shape = frames.get_shape().as_list()
        num_features = frame_shape[-1]
        
        batch = -1
        seq_length = self.kwargs['seq_length'] 
        if seq_length == 0 and len(frame_shape) != 3:
            batch = self.kwargs['batch_size']
        elif seq_length != 0:
            batch = seq_length * self.kwargs['batch_size']
            
        frames = tf.reshape(frames, (batch, num_features))
        
        return frames
    
    def get_model(self, frames):
        
        if 'audio' in self.kwargs['input_type'].lower():
            frames = self._reshape_to_conv(frames)
            audio = AudioModel(is_training=True).create_model(frames)
            output_model = self._reshape_to_rnn(audio)
        elif 'video' in self.kwargs['input_type'].lower():
            video = VideoModel(is_training=True).create_model(frames)
            output_model = self._reshape_to_rnn(video)
        
        rnn = RNNModel(self.kwargs['hidden_units']).create_model(output_model)
        
        if self.kwargs['seq_length'] == 0:
            rnn = rnn[:, -1, :]
        
        num_outputs = int(self.data_provider.label_shape[0])
        if self.kwargs['task'] == 'classification':
            num_outputs = self.data_provider.num_classes
        
        outputs = fully_connected(rnn, num_outputs)
        
        return outputs
    
    def start_process(self):
        if 'generate' in self.kwargs['which'].lower():
            generator_params = self._get_gen_params()
            g = UnimodalGenerator(**generator_params)
            g.write_tfrecords(self.kwargs['tfrecords_folder'])
            return
        
        self.data_provider = self.get_data_provider()
        predictions = self.get_model
        
        if 'train' in self.kwargs['which'].lower():
            train_params = self._get_train_params()
            train_params['predictions'] = predictions
            train_params['data_provider'] = self.data_provider
            
            TrainEval(**train_params).start_training()
        elif 'evaluate' in self.kwargs['which'].lower():
            eval_params = self._get_eval_params()
            eval_params['predictions'] = predictions
            eval_params['data_provider'] = self.data_provider
            
            SlimEvaluation(**eval_params).start_evaluation()
            eval_params = self._get_eval_params()

    def _get_train_params(self):
        train_params = {}
        train_params['train_dir'] = self.kwargs['train_dir']
        train_params['initial_learning_rate'] = self.kwargs['initial_learning_rate']
        train_params['num_epochs'] = self.kwargs['num_epochs']
        train_params ['loss'] = self.kwargs['loss']
        train_params['pretrained_model_checkpoint_path'] = \
            self.kwargs['pretrained_model_checkpoint_path']
        
        return train_params
    
    def _get_dp_params(self):
        dp_params = {}
        dp_params['tfrecords_folder'] = self.kwargs['tfrecords_folder']
        dp_params['seq_length'] = self.kwargs['seq_length']
        dp_params['batch_size'] = self.kwargs['batch_size']
        dp_params['task'] = self.kwargs['task']
        if dp_params['task'] == 'classification':
            dp_params['num_classes'] = self.kwargs['num_classes']
        
        return dp_params
    
    def _get_eval_params(self):
        eval_params = {}
        eval_params['train_dir'] = self.kwargs['train_dir']
        eval_params['log_dir'] = self.kwargs['log_dir']
        eval_params['seq_length'] = self.kwargs['seq_length']
        eval_params['metric'] = self.kwargs['metric'] #[x for x  in self.metric.split(',')]
        eval_params['eval_interval_secs'] = self.kwargs['eval_interval_secs']
        
        return eval_params
    
    def _get_gen_params(self):
        generator_params = {}
        file_reader = FileReader(self.kwargs['data_file'], delimiter=';')
        generator_params['input_type'] = self.kwargs['input_type']
        generator_params['reader'] = file_reader
        
        return generator_params
    
    def get_data_provider(self):
        dp_params = self._get_dp_params()
        if ('audio' or 'video') in self.kwargs['input_type']:
            provider = UnimodalProvider
        else:
            provider = MultimodalProvider
        
        data_provider = \
            provider(**dp_params)
        
        return data_provider

def main(_):
    
    flags = vars(parser.parse_args())
    with tf.Graph().as_default():
        e2u = End2You(**flags)
        e2u.start_process()

if __name__ == '__main__':
    tf.app.run()