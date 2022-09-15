# MobiusGCN
Internship at Labo ETIS  
A unofficial implement of the paper [3D Human Pose Estimation Using Möbius Graph Convolutional Networks](https://arxiv.org/pdf/2203.10554.pdf)
## Environment
This repository is build upon Python3.7 and PyTorchxxx. NVIDIA GPUs are needed to train and test. The other requirements are saved in the file [requirements.txt](https://github.com/paparazzi-z/MobiusGCN/blob/main/requirements.txt).
## Dataset
I setup the dataset following the method of [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). And you can find a clear instruction of dataset setup in the repository of [SemGCN](https://github.com/garyzhao/SemGCN/blob/master/data/README.md).  

The well setted dataset should have at least `data_2d_h36m_gt.npz` and `data_3d_h36m.npz`.
## Train
You can train the model by running the following command  

    python main_mobius.py --epochs 30 --num_layers 5 --hid_dim 128
`--epochs` is the number of epoch for training, `--num_layers` is the number of hidden layers except for the input layer and the output layer and `--hid_dim` is the number of channels of the model. More detailed `args` settings can be found in [main_mobius.py](https://github.com/paparazzi-z/MobiusGCN/blob/main/main_mobius.py).  

The model will be saved in the dirctory `checkpoint`.
## Test
One trained model can be found in the directory [checkpoint](https://github.com/paparazzi-z/MobiusGCN/tree/main/checkpoint/result).The model is trained 30 epoches with 128 channels. The ReLU operation has been removed to get the satisfying result. You can directly evaluate this model by running the following command

    python main_mobius.py --evaluate checkpoint/result/ckpt_best.pth.tar
`--evaluate` is the path of the evaluated model.
## Visualization
You can get the visualization result by running the following command

    python viz.py --evaluate checkpoint/result/ckpt_best.pth.tar --viz_subject S11 --viz_action Walking --viz_camera 0 --viz_output output.gif --viz_size 3 --viz_downsample 2 --viz_limit 60
`--evaluate`is the path of the evaluated model. `--viz_subject` is the group of dataset used. `--viz_action` is the action you want to visualize and you can find a full list of actions in [h36m_dataset.py](https://github.com/paparazzi-z/MobiusGCN/blob/main/common/h36m_dataset.py). `--viz_output` is the file name you want to save and the file will be saved at the root directory of the repository. More detailed `args` settings can be found in [viz.py](https://github.com/paparazzi-z/MobiusGCN/blob/main/viz.py).
## Acknowledgements
This work is mainly based on  
* [SemGCN](https://github.com/garyzhao/SemGCN)
* [CayleyNets](https://github.com/amoliu/CayleyNet)
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
