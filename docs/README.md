## Tutorials

We provide tutorials and a script [file](cli) to get you started with End2You. In particular we provide the following:

1. [Data Generation](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/1.%20Data%20Generation.ipynb)<br>
2. [Data Provider](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/2.%20Data%20Provider.ipynb)<br>
3. [Training Process](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/3.%20Training%20Process.ipynb)<br>
4. [Evaluation Process](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/4.%20Evaluation%20Process.ipynb)


## Models

We provide a number of models depending on the modality. Here we describe the audio, visual and multimodal modalities.

### Audio

We provide two models that can handle the audio modality. Both of them are papers published in ICASSP.

#### Emo18

The first one was published in 2018 and its architecture is shown below:

> `Tzirakis, P., Zhang, J., & Schuller, B. W. (2018, April). End-to-end speech emotion recognition using deep neural networks. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5089-5093). IEEE.`

![alt text](figures/emo18.png "Speech Emotion Recognition - Emo18 model")


#### Emo16

The second one was published in 2016 and its architecture is shown below:

> `Trigeorgis, G., Ringeval, F., Brueckner, R., Marchi, E., Nicolaou, M. A., Schuller, B., & Zafeiriou, S. (2016, March). Adieu features? end-to-end speech emotion recognition using a deep convolutional recurrent network. In 2016 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 5200-5204). IEEE..`

![alt text](figures/emo16.png "Speech Emotion Recognition - Emo16 model")


### Visual

For the visual modality we use the models from [`torchvision.models`](https://pytorch.org/docs/stable/torchvision/models.html) and provide the following ones:

1. ResNet - [18,34,50,101,152]
2. VGG - [11,13,16,19]
3. VGG_BN - [11,13,16,19]
4. DenseNet - [121,161,169,201]
5. MobileNet
6. ResNeXt - [32x4d, 32x8d]
7. Wide ResNet - [50, 101]
8. ShuffleNet - x[0.5, 1.0, 1.5, 2.0]

The input to the models should be of the size (96x96).

### Multimodal

We combine the audio and visual models in a multimodal one by fusing their features and feeding them in a RNN model for the final prediction.
An example of this type of model can be found in the following paper.

> `Tzirakis, P., Trigeorgis, G., Nicolaou, M. A., Schuller, B. W., & Zafeiriou, S. (2017). End-to-end multimodal emotion recognition using deep neural networks. IEEE Journal of Selected Topics in Signal Processing, 11(8), 1301-1309.`

![alt text](figures/multimodal.png "Multimodal Emotion Recognition model")
