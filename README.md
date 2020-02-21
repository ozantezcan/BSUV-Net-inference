# BSUV-Net: A Fully-Convolutional Neural Network forBackground Subtraction of Unseen Videos

This repository contains the source code of BSUV-Net algorithm as described in the following paper:

[BSUV-Net: A Fully-Convolutional Neural Network forBackground Subtraction of Unseen Videos](https://arxiv.org/pdf/1907.11371.pdf)
by M. Ozan Tezcan, Prakash Ishwar and Janusz Konrad.

BSUV-Net is a convolutional neural network which uses the concatenation of several 
images as input.
The descriptions of these images can be found in the paper.

Currently source code contains only the trained models and inference code that can be applied to any video.
`examples` folder includes an example video and its background subtraction result using
BSUV-Net with and without FPM.
## Requirements
1. [Python3](https://www.python.org/) (testen on v3.6)
2. [PyTorch](https://pytorch.org/) (tested on v1.3 and v1.4 with CUDA 10.1 on Ubuntu)
3. [OpenCV](https://opencv.org/releases/) for Python (tested on version 4.2.0)
4. [YACS](https://github.com/rbgirshick/yacs) (pip install yacs should work)

## Trained Models
Trained model which uses the RGB channels of empty background, recent background and
 current frame (9-channel input) can be downloaded at: 
https://drive.google.com/file/d/1q6a0RJuD54Gq8txw6TKPRRnC5xnSucLR/view?usp=sharing

Trained model which uses the RGB+FPM channels of empty background, recent background and 
current frame (12-channel input) can be downloaded at: 
https://drive.google.com/file/d/1ISzZyLDzuRuMnNmrZ3QVJCVeT_3eltDK/view?usp=sharing

## Usage
1. If you want to to use a model with FPM channel, download `HRNet_v2` model files from
http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/ and place them in
`utils/segmentation/hrnet_v2` folder.
2. Change `configs/infer_config.py` based on your application. 
3. Run `python inference.py <vid_in> <vid_out>` where `<vid_in>` is the path to the input video
and  `<vid_out>` is the desired output path.


