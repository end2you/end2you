# End2You - The Imperial Toolkit for Multimodal Profiling

We introduce End2You the Imperial toolkit for multimodal profiling. This repository provides easy-to-use scripts to train and evaluate either uni-modal or multi-modal models in an end-to-end manner for either regression or classification output. The input to the model can be one of the following:
- Visual Information : Face of a subject
- Speech Information : Speech waveform
- Audio-Visual Information : Face and speech of a subject

We use a ResNet with 50 layers to extract features from the visual information, while from the speech a 2-layer Convolutional Neural Network (CNN) is used. For the multimodal cases, we introduce a fully connected layer to map the features extracted from the different modalities to the same space. Afterwards, we have a 2-layer recurrent neural network and more particularly we utilise a Gated Recurrent Unit (GRU) to take into account the contextual information in the data.

<!--Pre-trained models are also provided for the visual, speech and multimodal cases. The pre-trained models were trained using the AVEC 2016 database (RECOLA).-->

### Citing

If you are using this toolkit please cite:

`Tzirakis, P.,Zafeiriou, S., & Schuller, B. (2017). End2You -- The Imperial Toolkit for Multimodal Profiling by End-to-End Learning. arXiv preprint arXiv:1802.01115.`

### Dependencies
Below are listed the required modules to run the code.

  * Python >= 3.4
  * NumPy >= 1.11.1
  * TensorFlow >= 1.4 (see ``Installation`` section for installing this module)
  * MoviePy >= 0.2.2.11
  * liac-arff >= 2.0
  * sklearn >= 0.19

### Pretrain Model

We provide a pretrained model of the ResNet-50 here:
https://www.doc.ic.ac.uk/~pt511/pretrain_model/model.ckpt-33604.zip

The model was trained on a non-publicly dataset from the [RealEyes](https://www.realeyesit.com/) Company.
Some statistics of the dataset can be found below:

| Attribute | Value |
| :---: | :---: |
| # Videos | 4,973 |
| # Frames | 1,059,505  |
| # Subjects | 2,616 |
| Age variation | [18-69] |
| # Emotions | 8 |
| # Annotators | 7 |

### Contents

1. [Installation](#installation)<br>
2. [Generating Data](#generating-data)<br>
3. [Training](#training)<br>
4. [Evaluation](#evaluation)<br>
5. [Testing](#testing)<br>
6. [Tutorial](#tutorial)<br>

## Installation
We highly recommended to use [conda](http://conda.pydata.org/miniconda.html) as your Python distribution.
Once downloading and installing [conda](http://conda.pydata.org/miniconda.html), this project can be installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n end2you python=3.5
$ source activate end2you
```

**Step 2:** Install [TensorFlow v.1.4](https://www.tensorflow.org/) following the 
official [installation instructions](https://www.tensorflow.org/install/install_linux#InstallingAnaconda). 
For example, for 64-bit Linux, the installation of GPU enabled, Python 3.5 TensorFlow involves:
```console
(end2you)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
(end2you)$ pip install --upgrade $TF_BINARY_URL
```

**Step 3:** Clone and install the `end2you` project as:
```console
(end2you)$ git clone git@github.com:end2you/end2you.git
```

## Setting the right flags

To run End2You, certain number of flags needs to be set. These are the following.

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --tfrecords_folder | The directory of the tfrecords files. | string | - |
| --input_type | Which model to consider. | audio, video, or both | - |
| --task | The task of the experiment. | classification<br />regression | classification |
| --num_classes | Only when task is classification. | int | 3 |
| --batch_size | The batch size to use. | int | 2 |
| --delimiter | The delimiter to use to read the files. | string | \t |
| --hidden_units | The number of hidden units in the RNN. | int | 128 |
| --num_rnn_layers | The number of layers in the RNN model. | int | 2 |
| --seq_length | The sequence length to introduce to the RNN.<br> If set to 0 indicates the whole raw file has a single label | int | 150 |
| <b>generate<br>train<br>evaluate<br>test</b> | What functionality to perform. | string | - |

## Generating Data

First, we need to convert the raw input data (audio, visual) in a format more suitable for
TensorFlow using TF Records. Both unimodal and multimodal inputs can be converted. 
To do so the user needs to provide a csv file that contains the full path of the raw data (e.g. .wav) and the label file for the data (default delimiter `\t`). The file needs to have the header "file;label". The `label` can be a file or a scalar. The scalar indicates that the whole file has a single label, while the file indicates that the raw data are split into segments with different labels.

> CSV File example - data_file.csv
```
file,label
/path/to/data/file1.wav,/path/to/labels/file1.csv
/path/to/data/file2.wav,/path/to/labels/file2.csv
```

The label file should contain a column with the timestep and the later columns with the label(s) of the timestep. The delimiter of this file should be the same as the delimiter of the `data_file.csv`. 

> Label File example - file1.csv
``` 
time,label1,label2
0.00,0.24,0.14
0.04,0.20,0.18
...
```

To create the tfrecords you need to specify the flag to be `generate`. An example is depicted below.

> Creating tf records
```console
(end2you)$ python main.py --tfrecords_folder=/where/to/save/tfrecords  \
                          --input_type=audio
                            generate  \
                          --data_file=data_file.csv \
```

By default the `tfrecords` will be generated in a folder called `tf_records` which 
contains the converted files of the `data_file`. To generate unimodal or multimodal input the `--input_type` need to be defined one of the following: `audio`, `video` or `audiovisual`. This operation takes one additional flag.

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --data_file | The path to the data file. | string | - |

## Training

To start training the model the `train` flag needs to be set. Two different training processes can start: 
1. Train only. This process will run for a user-defined number of epochs and performs only training. To evaluate a validation set you need to run separatelly the `Evaluation` process.
2. Train and evaluate model. The evaluation of the model is performed after each epoch, where also the model is saved in the `--train_dir` folder. Furthermore, the 5 best models are saved in the `--train_dir/top_k_models` along with the performance in the evaluation set in a file named `models_performance.txt`.

To start the second training process the `--tfrecords_eval_folder` flag needs to defined.

Both processes save logs and models. The training flags that can be defined are shown below.

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --train_dir | Directory where to write checkpoints and event logs. | string | ckpt/train |
| --initial_learning_rate | Initial learning rate. | float | 0.0001 |
| --loss | Which loss is going to be used: <br> Concordance Correlation Coefficient ('ccc') <br> Mean Squared Error ('mse') <br>  Softmax Cross Entropy ('sce') <br> Cross Entropy With Logits ('cewl') <br> | ccc, mse, sce, cewl | cewl |
| --pretrained_model_checkpoint_path | If specified, restore this pretrained model before beginning any training. | string | - |
| --num_epochs | The number of epochs to run training. | int | 50 |
| --seq_length | The sequence length to introduce to the RNN.<br> If set to 0 indicates the whole raw file has a single label | int | 150 |
| --batch_size | The batch size to use. | int | 2 |
| --tfrecords_folder | The directory of the tfrecords files. | string | - |
| --tfrecords_eval_folder | If specified, after each epoch evaluation of the model is performed during training. | string | - |
| --noise | Only for --input_type=audio. The random gaussian noise to introduce to the signal. | float | - |

> Example
```console
(end2you)$ python main.py --tfrecords_folder=path/to/tfrecords \
                          --input_type=audio \
                          train \
                          --train_dir=ckpt/train \
```

## Evaluation

***This evaluation should start when the `--tfrecords_eval_folder` flag is not set in the training.***

To start the evaluation of the model the parameter to be defined is `evaluate`. This script automatically evaluates a new model that is saved in the folder specified by the `--train_dir`, and it runs until the user manually stops it. The evaluation of the model is performed on the tfrecords files specified by the flag `--tfrecords_folder`. In addition, it is good practice to set the `--log_dir` flag to be saved in the same folder as in the train one. For example, if `--train_dir=ckpt/train` (set when executing the training script), then you can set `--log_dir=ckpt/log` (set when the evaluation script is executed).
If a flag is not specified in the execution command it will be initialised with the default value.

The following list of arguments can be used for evaluation.

| Flag | Description | Values | Default |
| :---: | :---: | :---: | :---: |
| --train_dir | Directory where to write checkpoints and event logs. | string | ckpt/train |
| --log_dir | Directory where to write event logs. | float | 0.0001 |
| --metric | Which metric to use for evaluation. One of: <br> Concordance Correlation Coefficient (ccc) <br> Mean Squared Error (mse) <br> Unweighted Average Recall (uar) | ccc, mse, uar | 'uar' |
| --eval_interval_secs | How often to run the evaluation (in sec). | int | 300 |


> Example
```console
(end2you)$ python main.py --tfrecords_folder=path/to/tfrecords \
                          --input_type=audio \
                            evaluate \
                          --train_dir=ckpt/train \
                          --log_dir=ckpt/log                          
```

**TensorBoard**: You can simultaneously run the training and validation. The results can be observed through TensorBoard. Simply run:

```
(end2you)$ tensorboard --logdir=ckpt
```

This makes it easy to explore the graph, data, loss evolution and performance on the validation set.

## Testing

***Currenntly only for raw audio files can be used. It will be extended for visual and audiovisual data.***

This process finds the predictions of a model on raw data files and saves them to disk.
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
| --data_file | The path of the test file. | string | - |
| --model_path | The model to test. | string | - |
| --prediction_file | The file to write predictions (in csv format) | string | predicitons.csv |

>  Get predictions - Example
```
python main.py --input_type=audio \ 
               --seq_length=0 \
               --task=classification \
               --num_classes=3 \
               test 
               --data_file=test_file.csv 
               --model_path=/path/to/model.ckpt-XXXX
```

## Tutorial

This tutorial provides examples to use the end2you toolkit. More particularly, you will learn:
  * Start training and evaluation of a uni-modal model.
  <!-- * Start training and evaluation of a multi-modal model.-->

### Start training and evaluation of a uni-modal model

In this example we will learn to train and evaluate a model using only the visual information of the data. To use the speech modality the `--model` flag needs to be set to `audio`. 
To start training the model the `main.py` script needs to be executed with the flag `--option=train`. An example is shown below.

> Example - start training
```console
(end2you)$ python main.py --tfrecords_folder=path/to/tfrecords \
                          --input_type=video
                          --batch_size=2 \
                          --seq_length=150                           
                            train \
                          --train_dir=ckpt/train \
```

To start evaluation of the model the `main.py` script needs to be executed with the flag `--option=evaluate`. Important flags are the following:
  * train_dir : where the training script is saving the checkpoints.
  * log_dir : the directory to save the log files.
  * portion : the portion (train, valid, test) to use to evaluate the model.
  * num_examples : number of examples in the PORTION set.

You need also to be certain that the flags to create the model, like `--num_gru_modules` and `--hidden_units`, to be the same as in the training script.

> Example - start evaluation
```console
(end2you)$ python main.py --tfrecords_folder=path/to/tfrecords \
                          --batch_size=1 \
                          --seq_length=150 
                          --input_type=video \
                            evaluate \
                          --train_dir=ckpt/train \
                          --log_dir=ckpt/log \    
```

If a flag is not specified, the default value is used.

#### Start training with a pre-trained model

If we want to start the training from a pre-trained model, like the one provided, we need to set the flag `pretrained_model_checkpoint_path`. For example, start training from a pre-trained video model.

> Example - start training using pre-trained model
```console
(end2you)$ python main.py --tfrecords_folder=path/to/tfrecords \
                          --input_type=audio \
                          --batch_size=2 \
                          --seq_length=150 \
                            train
                          --train_dir=ckpt/train \
                          --pretrained_model_checkpoint_path=path/to/model.ckpt-XXXX    
```

<!--
### Start training and evaluation of a multi-modal model

We provide two multi-modal cases, depending on the input, for training:
  1. Audio-Visual input. To use this option you need to define the `--model` flag to be equal to `audiovisual`.
  2. Audio-Visual-Physio input. To use this option you need to define the `--model` flag to be equal to `all`.

For example, suppose we want to start training using the audio-visual input:

> Example - start training with audio-visual input
```console
(end2you)$ python e2u_train.py --dataset_dir=path/to/tfrecords \
                               --train_dir=ckpt/train \
                               --model=audiovisual \
                               --batch_size=2 \
                               --seq_length=150 \
```

Evaluation of the model:

> Example - start evaluation with audio-visual input
```console
(end2you)$ python e2u_eval.py --dataset_dir=path/to/tfrecords \
                              --checkpoint_dir=ckpt/train \
                              --log_dir=ckpt/log \
                              --model=audiovisual \
                              --batch_size=1 \
                              --seq_length=150     
```


We can also start training using a pre-trained model, like the ones provided, by setting the `--pretrained_model_checkpoint_path` flag as explained in previous section.
-->
