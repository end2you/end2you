import argparse


def add_eval_args(parser):
    parser.add_argument('--valid_num_workers', type=int, default=1,
                        help='Number of workers to use (default `1`).')
    parser.add_argument('--valid_dataset_path', type=str, required=True,
                        help='The directory of the .hdf5 files.')
    parser.add_argument('--metric', type=str, default='mse',
                        help='Which metric to use for evaluation: ccc (Concordance Correlation Coefficient), '
                             'mse (Mean Squared Error), or uar (Unweighted Average Recall) (default `mse`).',
                        choices=['ccc', 'mse', 'uar'])
    return parser


def add_train_args(parser):
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (default `0.0001`).')
    parser.add_argument('--loss', type=str, default='mse',
                        help='Which loss is going to be used: ccc (Concordance Correlation Coefficient), '
                             'mse (Mean Squared Error), ce (Cross Entropy). '
                             '(default `mse`)',
                        choices=['ccc', 'mse', 'ce'])
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use during training (default `adam`).')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='The number of epochs to run training (default 50).')
    parser.add_argument('--train_num_workers', type=int, default=1,
                        help='Number of workers to use (default `1`).')
    parser.add_argument('--train_dataset_path', type=str, required=True,
                        help='The directory of the .hdf5 files.')
    parser.add_argument('--save_summary_steps', type=int, default=10,
                        help='Every which step to perform evaluation in training (default `10`).')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to checkpoint file.')
    
    return parser


def add_gen_args(parser):
    parser.add_argument('--save_data_folder', type=str, required=True,
                        help='Path to save `*.hdf5` files.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input file.')
    parser.add_argument('--delimiter', type=str, default=',',
                        help='Delimiter used to read input files (default `,`).')
    parser.add_argument('--exclude_cols', type=str, default=None,
                        help='Columns to exclude of the input files (default `None`).')
    parser.add_argument('--fieldnames', type=str, default=None,
                        help='''If no header exists in input files, one needs to specify it.
                                Header names are comma separated. Value of `None` indicates 
                                that a header exists (default `None`).''')
    
    return parser


def add_test_args(parser):
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='The folder path of files to test.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='The path of the model to test.')
    parser.add_argument('--prediction_file', type=str, default='predictions.csv',
                        help='The file to write predictions (in csv format).')
    parser.add_argument('--metric', type=str, default='mse',
                        help='Metric to use (defaule `mse`).')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers to use (default `1`).')
    return parser


def add_parsers():
    parser = argparse.ArgumentParser(description='End2You flags.')
    
    subparsers = parser.add_subparsers(help='Should be one of [generate, train, test].', dest='process')
    subparsers.required = True
    
    parser.add_argument('--modality', type=str, required=True,
                        help='Modality to use: audio, visual, or both.',
                        choices=['audio', 'visual', 'audiovisual'])
    parser.add_argument('--num_outputs', type=int, default=3,
                        help='Number of outputs of the model (default 2).')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The batch size to use. (default 2)')
    parser.add_argument('--seq_length', type=int, default=None,
                        help="""The sequence length to introduce to the RNN. 
                        If `None` it uses whole sequence as input (default `None`)""")
    parser.add_argument('--cuda', type=str, default='True',
                        help='Whether to use GPU or not (default `false`).')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use (default `1`).')
    parser.add_argument('--root_dir', type=str, default='./e2u_output',
                        help='Path to save models/results (default `./e2u_output`).')
    parser.add_argument('--log_file', type=str, default='./out_log.log',
                        help='Path to save log file (default `./out_log.log`).')
    parser.add_argument('--take_last_frame', type=str, default='false',
                        help='Consider the last frame of the signal to compute loss/metric (default `False`).')

    parser.add_argument('--model_name', type=str, default='resnet18',
                        help='Which visual model to use. (default `resnet18`).')
    parser.add_argument('--pretrained', type=str, default='false',
                        help='Whether to use pretrain (on ImageNet) visual model. (default `false`).')
    
    generation_subparser = subparsers.add_parser('generate', help='Generation arguments.')
    generation_subparser = add_gen_args(generation_subparser)
    
    training_subparser = subparsers.add_parser('train', help='Training argument options.')
    training_subparser = add_train_args(training_subparser)
    training_subparser = add_eval_args(training_subparser)
    
    test_subparser = subparsers.add_parser('test', help='Testing argument options.')
    test_subparser = add_test_args(test_subparser)
    
    return parser
