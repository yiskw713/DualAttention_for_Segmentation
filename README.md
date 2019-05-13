# DualAttention_for_Segmentation
This is the repository for re-implementation of dual attention network with pytorch. 

## Requirements
*python 3.x
* pytorch >= 1.0
* torchvision
* pandas
* numpy
* Pillow
* tqdm
* PyYAML
* addict
* tensorboardX
* adabound

## Dataset
### PASCAL VOC(2007/2012)
You can download from [this link](http://host.robots.ox.ac.uk/pascal/VOC/)


## Training
If you want to train a model, please run `python utils/build_dataset.py` to make csv_files for training and validation.

Then, just run `python train.py ./PATH_TO_CONFIG_FILE`

For example, when running `python train.py ./result/danet_drn_d_22/config.yaml`,
the configuration described in `./result/danet_drn_d_22/config.yaml` will be used .

If you want to set your own configuration, please make config.yaml like this:
```
model: drn_d_22
attention: True       # if you use dual attention modules or not

class_weight: True    # if you use class weight to calculate cross entropy or not
writer_flag: True     # if you use tensorboardx or not

n_classes: 21         # including background class
batch_size: 32
crop_height: 300
crop_width: 300
height: 256
width: 256
num_workers: 4
max_epoch: 300

optimizer: AdaBound
learning_rate: 0.001
lr_patience: 10       # Patience of LR scheduler
momentum: 0.9         # momentum of SGD
dampening: 0.0        # dampening for momentum of SGD
weight_decay: 0.001   # weight decay for SGD
nesterov: True        # enables Nesterov momentum

dataset_dir: /xxxx/xxxx/xxxx/VOCdevkit
year: 2012 # pascal voc 2007 or 2012
result_path: ./result/drn_d_22
```

## References
Dual Attention Network for Scene Segmentation, \
Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu, \
in CVPR2019\
  [arXiv](https://arxiv.org/pdf/1809.02983.pdf)\
  [Github](https://github.com/junfu1115/DANet)
