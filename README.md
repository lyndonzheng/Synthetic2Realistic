# Synthetic2Realistic
This repository implements the training and testing of T2Net for "T2Net: Synthetic-to-Realistic Translation for Depth Estimation Tasks" by Chuanxia Zheng, [Tat-Jen Cham](http://www.ntu.edu.sg/home/astjcham/) and [Jianfei Cai](http://www.ntu.edu.sg/home/asjfcai/) at NTU. The repository offers the original implementation of the paper in Pytoch.

<div style="max-width:640px; margin:0 auto 10px;" >
<div
style="position: relative;
width:100%;
padding-bottom:56.25%;
height:0;">
<iframe style="position: absolute;top: 0;left: 0;width: 100%;height: 100%;"  src="https://youtu.be/B6lOToIk0xY" frameborder="0" allowfullscreen></iframe>
</div>
</div>



This repository can be used for training and tesing of
- Unpaired image-to-image Translation
- Single depth Estimation

# Getting Started
## Installation
This code was tested with Pytoch 0.4.0, CUDA 8.0, Python 3.6 and Ubuntu 16.04
- Install Pytoch 0.4, torchvision, and other dependencies from [http://pytorch.org](http://pytorch.org)
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate) for visualization

```
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/lyndonzheng/Synthetic2Realistic
cd Synthetic2Realistic
```

## Datasets
The indoor Synthetic Dataset renders from [SUNCG](http://suncg.cs.princeton.edu/) and indoor Realistic Dataset comes from [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
The outdooe Synthetic Dataset is [vKITTI](http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds) and outdoor Realistic dataset is [KITTI](http://www.cvlibs.net/datasets/kitti/)

## Training
**Warning: The input sizes need to be muliples of 64. The feature GAN model needs to be change for different scale**

- Train a model with multi-domain datasets:

```
python train.py train.py --name Outdoor_nyu_wsupervised --model wsupervised
--img_source_file /dataset/Image2Depth31_KITTI/trainA_SYN80.txt
--img_target_file /dataset/Image2Depth31_KITTI/trainA.txt
--lab_source_file /dataset/Image2Depth31_KITTI/trainB_SYN80.txt
--lab_target_file /dataset/Image2Depth31_KITTI/trainB.txt
--shuffle --flip --rotation
```

- To view training results and loss plots, run python -m visdom.server and copy the URL [http://localhost:8097](http://localhost:8097).
- Training results will be saved under the *checkpoints* folder. The more training options can be found in *options*.

## Testing
- Test the model

```
python test.py --name Outdoor_nyu_wsupervised --model test
--img_source_file /export/home/lyndonzheng/dataset/Image2Depth31_KITTI/testA_SYN80
--img_target_file /export/home/lyndonzheng/dataset/Image2Depth31_KITTI/testA
```

## Trined Models

More trained models will be released

# Acknowledgments
Code is inspired by [Pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
