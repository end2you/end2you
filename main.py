import argparse

import tensorflow as tf

from src.models.audio_model import AudioModel
from src.models.video_model import VideoModel
from src.models.rnn_model import RNNModel
from src.models.base import fully_connected

from src.data_provider.unimodal_provider import UnimodalProvider 
from pathlib import Path
from src.training import Train
from src.evaluation import Eval
from src.data_generator.generate_unimodal import UnimodalGenerator
from src.data_generator.generate_multimodal import MultimodalGenerator
from src.parser import *

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
                         'if None (default) seq_length will be introduced' 
                         'by the audio model.')

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
        return tf.reshape(frames, (self.kwargs['batch_size'], 
                                   self.kwargs['seq_length'], 
                                   -1))
    
    def _reshape_to_conv(self, frames):
        return tf.reshape(frames, (self.kwargs['batch_size'] * 
                                   self.kwargs['seq_length'], 
                                   -1))
    
    def get_model(self, frames):
        if 'audio' in self.kwargs['input_type'].lower():
            audio = AudioModel(is_training=True).create_model(frames)
            output_model = self._reshape_to_rnn(audio)
        elif 'video' in self.kwargs['input_type'].lower():
            video = VideoModel(is_training=True).create_model(frames)
            output_model = self._reshape_to_rnn(video)
        
        rnn = RNNModel().create_model(output_model)
        
        rnn = self._reshape_to_conv(rnn)

        outputs = fully_connected(rnn, self.data_provider.label_shape)
        outputs = self._reshape_to_rnn(outputs)
        
        return outputs
    
    def start_process(self):
        if 'generate' in self.kwargs['which'].lower():
            generator_params = self._get_gen_params()
            g = UnimodalGenerator(**generator_params)
            g.write_tfrecords(self.kwargs['tfrecords_folder'])
            return
            
        self.data_provider = self.get_data_provider()
        frames, labels, sids = self.data_provider.get_batch()
        frames_shape = frames.get_shape().as_list()

        if len(frames_shape) == 3:
            frames = tf.reshape(frames, (-1, frames_shape[2]))
        
        predictions = self.get_model(frames)
        if 'train' in self.kwargs['which'].lower():
            train_params = self._get_train_params()
            train_params['predictions'] = predictions
            train_params['data_provider'] = self.data_provider

            Train(**train_params).start_training()
        elif 'evaluate' in self.kwargs['which'].lower():
            eval_params = self._get_eval_params()
            eval_params['predictions'] = predictions
            eval_params['data_provider'] = self.data_provider
            
            Eval(**eval_params).start_evaluation()
            eval_params = self._get_eval_params()

    
    def _get_train_params(self):
        train_params = {}
        train_params['input_type'] = self.kwargs['input_type']
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
        
        return dp_params
        
    def _get_eval_params(self):
        eval_params = {}
        eval_params['train_dir'] = self.kwargs['train_dir']
        eval_params['log_dir'] = self.kwargs['log_dir']
        eval_params['seq_length'] = self.kwargs['seq_length']
        eval_params['metric'] = self.kwargs['metric'] #[x for x  in self.metric.split(',')]
        eval_params['eval_interval_secs'] = self.kwargs['eval_interval_secs']
        
        num_examples = self.kwargs['num_examples']
        if self.kwargs['num_examples'] == None:
            num_examples = get_num_examples(self.kwargs['tfrecords_folder'])
        eval_params['num_examples'] = num_examples
        
        return eval_params
    
    def _get_gen_params(self):
        generator_params = {}
        generator_params['input_type'] = self.kwargs['input_type']
        generator_params['data_file'] = self.kwargs['data_file']
        
        return generator_params
    
    def get_data_provider(self):
        dp_params = self._get_dp_params()
        if 'audio' in self.kwargs['input_type'] or \
           'video' in self.kwargs['input_type']:
            provider = UnimodalProvider
        
        data_provider = \
            provider(**dp_params)
        
        return data_provider

def get_num_examples(tfrecords_folder):
    root_folder = Path(tfrecords_folder)
    num_examples = 0
    for tf_file in root_folder.glob('*.tfrecords'):
        for record in tf.python_io.tf_record_iterator(str(tf_file)):
            num_examples += 1

    return num_examples

def main(_):
    
    flags = vars(parser.parse_args())
    with tf.Graph().as_default():
        e2u = End2You(**flags)
        e2u.start_process()

if __name__ == '__main__':
    tf.app.run()