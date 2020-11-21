## Setting the right flags

We provide a script file that can either `generate` the hdf5 files, `train` one of our defined models, or `test` a model on new data.
To run it, certain number of flags needs to be set. These are the following.

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --modality | Modality to use. | audio, visual, or audiovisual | - |
| --num_outputs | Number of outputs of the model. | int | 2 |
| --batch_size | The batch size to use. | int | 2 |
| --delimiter | The delimiter to use to read the files. | str | , |
| --seq_length | The sequence length to introduce to the RNN.<br> if `None` the whole sequence will be considered.| int | None |
| --cuda | Whether to use cuda. | bool | False |
| --root_dir | Path to save models/results. | str | `./e2u_output` |
| <b>generate<br>train<br>test</b> | What process to perform. | string | - |

## Generating Data

First, we need to convert the raw input data (audio, visual) in a format more suitable for training/evaluating the models using `.hdf5` format. Both unimodal and multimodal inputs can be converted. 
To do so the user needs to provide a csv file that contains the full path of the raw data (e.g. `.wav`) and the label file for the data (default delimiter `,`). The file needs to have the header "file,label_file". 

> CSV File example - input_file.csv
```
file,label_file
/path/to/data/file1.wav,/path/to/labels/file1.csv
/path/to/data/file2.wav,/path/to/labels/file2.csv
```

The `label_file` is a file containing a header of the form `timestamp,label_1,label_2`, where `timestamp` is the time segment of the raw sequence with the corresponding labels (e.g. `label_1, label_2,...`).

> Label File example - file1.csv
``` 
time,label1,label2
0.00,0.24,0.14
0.04,0.20,0.18
...
```

To create the hdf5 file you need to specify the flag to be `generate`. Two flags are required for 

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --save_data_folder | Path to save `*.hdf5` files. | str | - |
| --input_file | Path to the input csv file. | str | - |
| --delimiter | Delimiter used to read input files. | str | , |
| --exclude_cols | Columns to exclude of the input files. | str | None |
| --fieldnames | If no header exists in input files, one needs to specify it. <br> Header names are comma separated. Value of `None` indicates <br> that a header exists | string | None |

An example is shown below.

> Creating `.hdf5` files
```console
(end2you)$ python main.py --modality=visual \
                          generate \
                          --save_data_folder='/path/to/save/hdf5/files' \
                          --input_file='/path/to/input_filecsv'
```


## Training

To start training the model the `train` flag needs to be set. Evaluation of the model is performed after each training epoch using `validation` data. During training a log file saves training information including epoch, loss, metric score etc. In addition, summary writers for tensorboard visualisation are stored. 
The flags that can be used are the following:

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --learning_rate | Initial learning rate. | float | 0.0001 |
| --loss | Which loss is going to be used: <br> Concordance Correlation Coefficient ('ccc') <br> Mean Squared Error ('mse') <br>  Softmax Cross Entropy ('sce') <br> | ccc, mse, ce | mse |
| --model_path | If specified, restore this pretrained model before beginning any training. | string | - |
| --num_epochs | The number of epochs to run training. | int | 50 |
| --batch_size | The batch size to use. | int | 2 |
| --train_dataset_path | The directory of the training files. | str | - |
| --valid_dataset_path | The directory of the training files. | str | - |
| --train_num_workers | Number of workers to use for fetching training data. | int | 1 |
| --valid_num_workers | Number of workers to use for fetching training data. | int | 1 |
| --log_file | Path to save log file. | str | - |
| --metric | Metric to use to evaluate model: <br> Concordance Correlation Coefficient ('ccc') <br> Mean Squared Error ('mse') <br> Unweighted Average Recall ('uar') <br> | str | mse |
| --save_summary_steps | Every which step to perform evaluation in training. | int | 10 |

> Example
```console
(end2you)$ python main.py --modality=visual \
                          train \
                          --train_dataset_path='/path/to/training/data' \
                          --valid_dataset_path='/path/to/validation/data' 

```

## Testing

This process finds the predictions of a model.
To begin with, a file with the path to these files needs to be created, with the header to contain the text `file`. An example is shown below.

> CSV File example - test_file.csv
```
file
/path/to/data/file1.wav
/path/to/data/file2.wav
```

Then the following flags needs to be defined.

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --dataset_path | Path with hdf5 files to run testing. | str | - |
| --model_path | The model to test. | str | - |
| --prediction_file | The file to write predictions (in csv format) | str | predictions.csv |
| --metric | Metric to use to evaluate model: <br> Concordance Correlation Coefficient ('ccc') <br> Mean Squared Error ('mse') <br> Unweighted Average Recall ('uar') <br> | str | mse |
| --num_workers | Number of workers to use for fetching the data. | int | 1 |

>  Get predictions - Example
```console
(end2you)$ python main.py --modality=visual \
                          test \
                          --model_path='/path/to/model.pth.tar' \
                          --dataset_path='/path/to/test/folder' 
```
