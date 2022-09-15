# MobiusGCN
Internship at Labo ETIS  
A no official implement of the paper [3D Human Pose Estimation Using Möbius Graph Convolutional Networks](https://arxiv.org/pdf/2203.10554.pdf)
## Environment
This repository is build upon Python3.7 and PyTorchxxx. The other requirements are saved in the file [requirements.txt](https://github.com/paparazzi-z/MobiusGCN/blob/main/requirements.txt).
## Dataset
I setup the dataset following the method of [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). And you can find a clear instruction of dataset setup in the repository of [SemGCN](https://github.com/garyzhao/SemGCN/blob/master/data/README.md).
## Train
You can train the model by running the following command  

    python main_mobius.py --epochs 30 --num_layers 5 --hid_dim 128
`--epochs` is the number of epoch for training, `--num_layers` is the number of hidden layers except for the input layer and the output layer and `--hid_dim` is the number of channels of the model. More detailed `args` settings can be found in [main_mobius.py](https://github.com/paparazzi-z/MobiusGCN/blob/main/main_mobius.py).
## Test
