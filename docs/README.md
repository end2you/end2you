## Tutorials

We provide tutorials to get you started with End2You. In particular we provide the following:

1. [Data Generation](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/1.%20Data%20Generation.ipynb)<br>
2. [Data Provider](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/2.%20Data%20Provider.ipynb)<br>
3. [Training Process](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/3.%20Training%20Process.ipynb)<br>
4. [Evaluation Process](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/tutorials/4.%20Evaluation%20Process.ipynb)


## Models

### Audio

We provide two models that can handle the audio modality. Both of them are papers published in ICASSP.
The first one was published in 2018 and its architecture is shown below:

> `Tzirakis, P., Zhang, J., & Schuller, B. W. (2018, April). End-to-end speech emotion recognition using deep neural networks. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5089-5093). IEEE.`

![alt text](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/figures/emo18.png "Speech Emotion Recognition - Emo18 model")

The second one was published in 2018 and its architecture is shown below:

> `Trigeorgis, George, Fabien Ringeval, Raymond Brueckner, Erik Marchi, Mihalis A. Nicolaou, Björn Schuller, and Stefanos Zafeiriou. "Adieu features? end-to-end speech emotion recognition using a deep convolutional recurrent network." In 2016 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 5200-5204. IEEE, 2016.`

![alt text](https://github.com/end2you/end2you/blob/end2you_pytorch/docs/figures/emo16.png "Speech Emotion Recognition - Emo16 model")


### Visual

For the visual modality we use the models from `torchvision.models` and provide the following ones:
{
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
            
            'vgg11': 25088,
            'vgg13': 25088,
            'vgg16': 25088,
            'vgg19': 25088,
            
            'vgg11_bn': 25088,
            'vgg13_bn': 25088,
            'vgg16_bn': 25088,
            'vgg19_bn': 25088,
            
            'densenet121': 9216,
            'densenet169': 14976,
            'densenet161': 19872,
            'densenet201': 17280,
            
            'mobilenet_v2': 11520,
            
            'resnext50_32x4d': 2048,
            'resnext101_32x8d': 2048,
            
            'wide_resnet50_2': 2048,
            'wide_resnet101_2': 2048,
            
            'shufflenet_v2_x0_5': 9216,
            'shufflenet_v2_x1_0': 9216,
            'shufflenet_v2_x1_5': 9216,
            'shufflenet_v2_x2_0': 18432
}
