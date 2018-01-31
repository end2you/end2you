# End2You - The Imperial Toolkit for Multimodal Profiling

We introduce End2You the Imperial toolkit for multimodal profiling. This repository provides easy-to-use scripts to train and evaluate either uni-modal or multi-modal models in an end-to-end manner for either regression or classification output. The input to the model can be one of the following:
- Visual Information : Face of a subject
- Speech Information : Speech waveform
- Audio-Visual Information : Face and speech of a subject

We use a ResNet with 50 layers to extract features from the visual information, while from the speech a 2-layer Convolutional Neural Network (CNN) is used. For the multimodal cases, we introduce a fully connected layer to map the features extracted from the different modalities to the same space. Afterwards, we have a 2-layer recurrent neural network and more particularly we utilise a Gated Recurrent Unit (GRU) to take into account the contextual information in the data.

Pre-trained models are also provided for the visual, speech and multimodal cases. The pre-trained models were trained using the AVEC 2016 database (RECOLA).

### Citing

If you are using this toolkit please cite:

`Tzirakis, P., Trigeorgis, G., Nicolaou, M. A., Schuller, B., & Zafeiriou, S. (2017). End-to-End Multimodal Emotion Recognition using Deep Neural Networks. arXiv preprint arXiv:1704.08619.`

### Dependencies
Below are listed the required modules to run the code.

  * Python >= 3.4
  * NumPy >= 1.11.1
  * TensorFlow >= 1.4 (see ``Installation`` section for installing this module)
  * MoviePy >= 0.2.2.11
  * liac-arff >= 2.0

### Contents

1. [Installation](#installation)<br>
2. [Generating Data](#generating-data)<br>
3. [Training](#training)<br>
4. [Evaluation](#evaluation)<br>
5. [Tutorial](#tutorial)<br>

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
| --input_type | Which model to consider. | audio, video, or both | - |
| --task | The task of the experiment. | classification<br />regression | classification |
| --num_classes | Only when task is classification. | int | 3 |
| --hidden_units | The number of hidden units in the RNN. | int | 128 |
| --num_rnn_layers | The number of layers in the RNN model. | int | 2 |
| --seq_length | The sequence length to introduce to the RNN.<br> If set to 0 indicates the whole raw file has a single label | int | 150 |
| --batch_size | The batch size to use. | int | 2 |
| --tfrecords_folder | The directory of the tfrecords files. | string | - |
| --delimiter | The delimiter to use to read the files. | string | \t |
| <b>generate<br>train<br>evaluate<br>test</b> | What functionality to perform. | string | \t |

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
contains the converted files of the `data_file`. To generate unimodal or multimodal input the `--input_type` need to be defined one of the following: `audio`, `video` or `audiovisual`.

## Training

To start training the model the `option` parameter of the main script needs to be set to `train`. The script also saves periodically checkpoints of the model.
For training the following list of arguments can be defined.

```
  --tfrecords_folder TFRECORDS_FOLDER
                        The tfrecords directory.
                        (default 'path/to/tfrecords')
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate. 
                        (default 0.0001)
  --batch_size BATCH_SIZE
                        The batch size to use. 
                        (default 2)
  --seq_length SEQ_LENGTH     
                        The sequence length to unfold the recurrent model. 
                        (default 150)                    
  --hidden_units HIDDEN_UNITS
                        The number of hidden units in the recurrent model. 
                        (default 128)
  --train_dir TRAIN_DIR
                        Directory where to write event logs and checkpoint. 
                        (default 'ckpt/train')
  --pretrained_model_checkpoint_path PRETRAINED_MODEL_CHECKPOINT_PATH
                        If specified, restore this pretrained model before
                        beginning any training. 
                        (default None)
  --num_epochs NUM_EPOCHS
                        Number of epochs to train model. 
                        (default 1)
  --loss LOSS
                        Loss to train model.
                        (default 'CCC')
  --input_type INPUT_TYPE         
                        Which model is going to be used: audio, video, audiovisual or all. 
                        (default 'video')

```

> Example
```console
(end2you)$ python main.py --tfrecords_folder=path/to/tfrecords \
                          --input_type=audio \
                          train \
                          --train_dir=ckpt/train \
```

The most important flag is the `--tfrecords_folder` which specifies the directory where the tfrecords are saved. If a flag is not specified during execution, it will be initialised with the default value.

## Evaluation

To start the evaluation of the model the parameter to be defined is `evaluate`.
The following list of arguments can be used for evaluation.

```
  --batch_size BATCH_SIZE
                        The batch size to use.
                        (default 1)
  --num_gru_modules     
                        How many GRU modules to use.
                        (default 2)
  --dataset_dir DATASET_DIR
                        The tfrecords directory.
                        (default 'path/to/tfrecords')
  --train_dir TRAIN_DIR
                        The directory that contains the saved model.
                        (default 'ckpt/train')
  --log_dir LOG_DIR
                        The directory to save log files.
                        (default 'ckpt/log')
  --num_examples NUM_EXAMPLES
                        Number of examples in the PORTION set.
                        (default 1)
  --model MODEL         
                        Which model is going to be used: audio,video, or both
                        (default 1)
  --hidden_units HIDDEN_UNITS
                        The number of hidden units in the recurrent model.
                        (default 128)
  --portion PORTION     
                        Dataset portion to use for training (train or devel).
                        (default 'valid')
  --seq_length SEQ_LENGTH     
                        The sequence length to unfold the recurrent model.
                        (default 150)
  --eval_interval_secs EVAL_INTERVAL_SECS     
                        How often to run the evaluation (in sec).
                        (default 300)
```

> Example
```console
(end2you)$ python main.py --tfrecords_folder=path/to/tfrecords \
                          --input_type=audio \
                            evaluate \
                          --train_dir=ckpt/train \
                          --log_dir=ckpt/log                          
```

The most important flags are the `--tfrecords_folder`, which specifies the directory where the tfrecords are saved, and the `--train_dir` which specifies where the training script saves checkpoints (this flag should be equal to the `--train_dir` flag specified when the training process started).  In addition, it is good practice to set the `--log_dir` flag to be saved in the same folder as in the train one. For example, if `--train_dir=ckpt/train` (set when executing the training script), then you can set `--log_dir=ckpt/log` (set when the evaluation script is executed).
If a flag is not specified in the execution command it will be initialised with the default value.

**TensorBoard**: You can simultaneously run the training and validation. The results can be observed through TensorBoard. Simply run:

```
(end2you)$ tensorboard --logdir=ckpt
```

This makes it easy to explore the graph, data, loss evolution and accuracy on the validation set. Once you have a models which performs well on the evaluation set you can stop the training process.

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
