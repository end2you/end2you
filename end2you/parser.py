import argparse


def add_eval_args(parser):
    parser.add_argument('--log_dir', type=str, default='ckpt/log',
                        help='Directory where to write event logs (for evaluation).')
    parser.add_argument('--metric', type=str, default='uar',
                        help='Which metric to use for evaluation: ccc (Concordance Correlation Coefficient), '
                             'mse (Mean Squared Error), or uar (Unweighted Average Recall).',
                        choices=['ccc', 'mse', 'uar'])
    parser.add_argument('--eval_interval_secs', type=int, default=300,
                        help='How often to run the evaluation (in sec).')
    parser.add_argument('--train_dir', type=str, default='ckpt/train',
                        help='Directory where to write checkpoints and event logs.')
    
    return parser

def add_train_args(parser):
    parser.add_argument('--train_dir', type=str, default='ckpt/',
                        help='Directory where to write checkpoints and event logs.')
    parser.add_argument('--initial_learning_rate', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--loss', type=str, default='cewl',
                        help='Which loss is going to be used: ccc (Concordance Correlation Coefficient), '
                             'mse (Mean Squared Error), sce (Softmax Cross Entropy), or '
                             'cewl (Cross Entropy With Logits). (default cewl)',
                        choices=['ccc', 'mse', 'sce', 'cewl'])
    parser.add_argument('--pretrained_model_checkpoint_path', type=str, default='',
                        help='If specified, restore this pretrained model'
                             'before beginning any training.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='The number of epochs to run training (default 50).')
    parser.add_argument('--tfrecords_eval_folder', type=str, default=None,
                        help='If specified, after each epoch evaluation of the model is'
                             'performed during training. (default None).')
    parser.add_argument('--noise', type=float, default=None,
                        help='Only for --input_type=audio. The random gaussian noise to introduce '
                              'to the signal. (default None).')
    
    return parser

def add_gen_args(parser):
    parser.add_argument('--data_file', type=str, required=True,
                        help='The files to convert to tfrecords.')
    parser.add_argument('--upsample', type=bool, default=False,
                        help='Whether to upsample data (can be used when number of labels is one).')
    
    return parser

def add_test_args(parser):
    parser.add_argument('--data_file', type=str, required=True,
                        help='Includes the path of files to test.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='The full path of the model to test.')
    parser.add_argument('--prediction_file', type=str, default='predictions.csv',
                        help='The file to write predictions (in csv format).')
    
    return parser
