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

slim = tf.contrib.slim

# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('option', 'evaluate',
                           'One of: train, evaluate, generate.')
tf.app.flags.DEFINE_string('tfrecords_folder', 'path/to/tfrecords',
                           'The tfrecords directory.')
tf.app.flags.DEFINE_string('split_name', 'test',
                           'The split to take for evaluation.')
tf.app.flags.DEFINE_integer('seq_length', 150, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', 2, '''The batch size to use.''')
tf.app.flags.DEFINE_string('frame_shape', '32',
                           'Define the shape of frames as saved in the data_generator')
tf.app.flags.DEFINE_string('label_shape', '2',
                           'Define the shape of labels as saved in the data_generator')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('log_dir', 'ckpt/log',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('input_type', 'audio',
                           '''Which model is going to be used: audio, video, or both ''')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of batches to run.')
tf.app.flags.DEFINE_string('loss', 'ccc',
                           '''Which loss is going to be used: ccc, or mse ''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
			               '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_integer('hidden_units', 128, 'Number of batches to run.')

# FLAGS for Evaluation class
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('portion', 'test', 'The number of examples in the test set')
tf.app.flags.DEFINE_string('data_file', '../data_file.csv', 'The files to create tfrecords')
tf.app.flags.DEFINE_integer('num_examples', 67500, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('metric', 'ccc',
                           '''Which loss is going to be used: ccc, or mse ''')

class End2You:
    
    def __init__(self, *args, **kwargs):
        
        self.tfrecords_folder = Path(kwargs['tfrecords_folder'])
        self.split_name = kwargs['split_name']
        self.seq_length = kwargs['seq_length']
        self.batch_size = kwargs['batch_size']
        self.frame_shape = kwargs['frame_shape']
        self.label_shape = kwargs['label_shape']
        
        self.train_dir = Path(kwargs['train_dir'])
        self.log_dir = Path(kwargs['log_dir'])
        self.input_type = kwargs['input_type']
        
        self.initial_learning_rate = kwargs['initial_learning_rate']
        self.num_epochs = kwargs['num_epochs']
        self.loss = kwargs['loss'].lower()
        self.pretrained_model_checkpoint_path = \
                            kwargs['pretrained_model_checkpoint_path']
        
        self.eval_interval_secs = kwargs['eval_interval_secs']
        self.portion = kwargs['portion']
        self.num_examples = kwargs['num_examples']
        self.metric = kwargs['metric']
        self.data_file = kwargs['data_file']
        
    def _reshape_to_rnn(self, frames):
        return tf.reshape(frames, (self.batch_size, 
                                   self.seq_length, 
                                   -1))
    
    def _reshape_to_conv(self, frames):
        return tf.reshape(frames, (self.batch_size * 
                                   self.seq_length, 
                                   -1))
    
    def get_model(self, frames):
        if 'audio' in self.input_type.lower():
            audio = AudioModel(is_training=True).create_model(frames)
            output_model = self._reshape_to_rnn(audio)
        elif 'video' in self.input_type.lower():
            video = VideoModel(is_training=True).create_model(frames)
            output_model = self._reshape_to_rnn(video)
        
        rnn = RNNModel().create_model(output_model)
        rnn = self._reshape_to_conv(rnn)

        outputs = fully_connected(rnn, 2)
        outputs = self._reshape_to_rnn(outputs)
        
        return outputs
    
    def start_training(self):
        data_provider = self.get_data_provider()
        frames, labels, sids = data_provider.get_batch()
        frames_shape = frames.get_shape().as_list()
        
        if len(frames_shape) == 3:
            frames = tf.reshape(frames, (-1, frames_shape[2]))
        
        predictions = self.get_model(frames)
        train_params = self._get_train_params()
        train_params['predictions'] = predictions
        train_params['data_provider'] = data_provider
        
        Train(**train_params).start_training()
    
    def start_evaluation(self):
        data_provider = self.get_data_provider()
        frames, labels, sids = data_provider.get_batch()
        frames_shape = frames.get_shape().as_list()
        
        if len(frames_shape) == 3:
            frames = tf.reshape(frames, (-1, frames_shape[2]))
        
        predictions = self.get_model(frames)
        eval_params = self._get_eval_params()
        eval_params['predictions'] = predictions
        eval_params['data_provider'] = data_provider
        
        Eval(**eval_params).start_evaluation()
    
    def start_generate(self):
        generator_params = self._get_gen_params()
        g = UnimodalGenerator(**generator_params)
        g.write_tfrecords(self.tfrecords_folder)
    
    def _get_train_params(self):
        train_params = {}
        train_params['input_type'] = self.input_type
        train_params['train_dir'] = self.train_dir
        train_params['initial_learning_rate'] = self.initial_learning_rate
        train_params['num_epochs'] = self.num_epochs
        train_params['loss'] = self.loss
        train_params['pretrained_model_checkpoint_path'] = \
            self.pretrained_model_checkpoint_path
        
        return train_params
                 
    def _get_dp_params(self):
        dp_params = {}
        dp_params['tfrecords_folder'] = self.tfrecords_folder
        dp_params['split_name'] = self.split_name
        dp_params['frame_shape'] = self.frame_shape
        dp_params['label_shape'] = self.label_shape
        dp_params['seq_length'] = self.seq_length
        dp_params['batch_size'] = self.batch_size
        
        return dp_params
        
    def _get_eval_params(self):
        eval_params = {}
        eval_params['input_type'] = self.input_type
        eval_params['train_dir'] = self.train_dir
        eval_params['log_dir'] = self.log_dir
        eval_params['tfrecords_folder'] = self.tfrecords_folder
        eval_params['split_name'] = self.split_name
        eval_params['seq_length'] = self.seq_length
        eval_params['metric'] = [x for x  in self.metric.split(',')]
        eval_params['eval_interval_secs'] = self.eval_interval_secs
        eval_params['portion'] = self.portion
        eval_params['num_examples'] = self.num_examples
        eval_params['num_outputs'] = self.label_shape
        eval_params['seq_length'] = self.seq_length
        
        return eval_params
    
    def _get_gen_params(self):
        generator_params = {}
        generator_params['input_type'] = self.input_type
        generator_params['data_file'] = self.data_file
        
        return generator_params
    
    def get_data_provider(self):
        dp_params = self._get_dp_params()
        if 'audio' in self.input_type or \
           'video' in self.input_type:
            provider = UnimodalProvider
        
        data_provider = \
            provider(**dp_params)
        
        return data_provider
    
def flags_to_dict():
    def _get_shape(shape):
        return [int(x) for x  in shape.split(',')]
    
    return {
        'tfrecords_folder': FLAGS.tfrecords_folder,
        'split_name': FLAGS.split_name,
        'seq_length': FLAGS.seq_length,
        'batch_size': FLAGS.batch_size,
        'frame_shape': _get_shape(FLAGS.frame_shape),
        'label_shape': _get_shape(FLAGS.label_shape),
        'train_dir': FLAGS.train_dir,
        'log_dir': FLAGS.log_dir,
        'input_type': FLAGS.input_type,
        'initial_learning_rate': FLAGS.initial_learning_rate,
        'num_epochs': FLAGS.num_epochs,
        'pretrained_model_checkpoint_path': FLAGS.pretrained_model_checkpoint_path,
        'loss': FLAGS.loss,
        'eval_interval_secs': FLAGS.eval_interval_secs,
        'portion': FLAGS.portion,
        'num_examples': FLAGS.num_examples,
        'metric': FLAGS.metric,
        'data_file': FLAGS.data_file        
    }

def main(_):
    flags = flags_to_dict()
    e2u = End2You(**flags)
    if 'evaluate' in FLAGS.option.lower():
        e2u.start_evaluation()
    elif 'train' in FLAGS.option.lower():
        e2u.start_training()
    elif 'generate' in FLAGS.option.lower():
        e2u.start_generate()
    else:
        raise ValueError('Option flag should be either evaluate or train') 
    
if __name__ == '__main__':
    tf.app.run()