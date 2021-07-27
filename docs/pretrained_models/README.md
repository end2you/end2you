# Pretrained Models (constantly updating)

We provide pretrained models in a number of databases and modalities. To reproduce the results we provide notebooks to train/evaluate models. 
Currently, the following emotion recognition databases are supported:

* [AffWild](AffWild)
* [RECOLA](RECOLA)
* [SEWA](SEWA)

## SEWA

The following tables provide results on the SEWA dataset (in terms of CCC) for the prediction of arousal and valence. In parenthesis are the performances obtained on the development set. Also the link to the pretrain models is provided.

### Visual modality

| Model | Arousal | Valence | link |
| :---: | :---: | :---: | :---: |
| ResNet18  | 0.484 (0.544) | 0.591 (0.655) | [download](https://www.doc.ic.ac.uk/~pt511/end2you_models/sewa/resnet18.pth.tar) |
| ResNet34  | 0.567 (0.584) | 0.612 (0.657) | [download](https://www.doc.ic.ac.uk/~pt511/end2you_models/sewa/resnet34.pth.tar) |
| ResNet50  | 0.620 (0.641) | 0.670 (0.639) | [download](https://www.doc.ic.ac.uk/~pt511/end2you_models/sewa/resnet50.pth.tar) |
| MobileNet | 0.433 (0.622) | 0.628 (0.639) | [download](https://www.doc.ic.ac.uk/~pt511/end2you_models/sewa/mobilenet_v2.pth.tar) |


## RECOLA

### Audio modality

The following tables provide results on the RECOLA dataset (in terms of CCC) for the prediction of arousal and valence. In parenthesis are the performances obtained on the development set. Also the link to the pretrain models is provided.

| Model | Arousal | Valence | link |
| :---: | :---: | :---: | :---: |
| Emo16   | - (0.623) | - (0.323) | [download](https://www.doc.ic.ac.uk/~pt511/end2you_models/recola/emo16.pth.tar) |
| Zhao19  | - (0.764) | - (0.431) | [download](https://www.doc.ic.ac.uk/~pt511/end2you_models/recola/zhao19.pth.tar) |
