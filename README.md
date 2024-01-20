# Hard-Label based Small Query Black-box Adversarial Attack
Pytorch code for WACV 2024 paper [Hard-Label based Small Query Black-box Adversarial Attack](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Hard-Label_Based_Small_Query_Black-Box_Adversarial_Attack_WACV_2024_paper.pdf).
This repository contains the demo code for reproducing the experimental results of SQBA. SQBA is a transferable model embedded hard-label black-box adversarial attack method. 

# Dependencies
- pytorch
- numpy

# Pretrained Model
As cited in the paper, networks and their weights can be downloaded from (https://github.com/huyvnphan/PyTorch_CIFAR10?tab=readme-ov-file)

# Setup and run
- Create "model/cifar10" and "data" directories for weights and dataset respectively
- Download weights resnet18.pt and mobilenet_v2.pt
- Download cifar10 dataset
- run attack_tb.py

# Citation
If you use SQBA for your research, please cite the paper:
```
@Article{SQBA,
  author  = {J. Park and N. McLaughlin and P. Miller},  
  journal = {IEEE/CVF Winter Conference on Application of Computer Vision},
  title   = {Hard-Label based Small Query Black-box Adversarial Attack},
  year    = {2024},
}
```
