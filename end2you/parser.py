import argparse


def add_eval_args(parser):
    parser.add_argument('--log_dir', type=str, default='ckpt/log',
                        help='Directory where to write event logs (for evaluation).')
    parser.add_argument('--num_examples', type=int,
                        help='The number of examples in the set that is evaluated;'
                             'if None program will find this value.')
    parser.add_argument('--num_outputs', type=int,
                        help='The number of outputs to be predicted by the model.')
    parser.add_argument('--metric', type=str, default='ccc',
                        help='Which loss is going to be used: ccc, mse or uar.',
                        choices=['ccc', 'mse', 'uar'])
    parser.add_argument('--eval_interval_secs', type=int, default=300,
                        help='How often to run the evaluation (in sec).')
    parser.add_argument('--train_dir', type=str, default='ckpt/train',
                        help='Directory where to write checkpoints and event logs.')
    
    return parser

def add_train_args(parser):
    parser.add_argument('--train_dir', type=str, default='ckpt/train',
                        help='Directory where to write checkpoints and event logs.')
    parser.add_argument('--initial_learning_rate', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--loss', type=str, default='ccc',
                        help='Which loss is going to be used: ccc, mse, or ce.',
                        choices=['ccc', 'mse', 'ce'])
    parser.add_argument('--pretrained_model_checkpoint_path', type=str,
                        help='If specified, restore this pretrained model'
                             'before beginning any training.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='The number of epochs to run training (default 10).')
    
    return parser

def add_gen_args(parser):
    parser.add_argument('--data_file', type=str,
                        help='The files to convert to tfrecords')
    
    return parser
