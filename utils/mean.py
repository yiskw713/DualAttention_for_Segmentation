def get_mean(norm_value=255):
    # calculated from PASCAL VOC 2012 train data
    return [
        116.4832 / norm_value, 112.9989 / norm_value, 104.1170 / norm_value
    ]


def get_std(norm_value=255):
    # calculated from PASCAL VOC 2012 train data
    return [
        60.4114 / norm_value, 59.4824 / norm_value, 60.9276 / norm_value
    ]


"""
If you want to calculate the mean and the std by your own,
please run the below code:


```
import yaml
import numpy as np
import torch

from addict import Dict

from utils.dataset import PASCALVOC
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.transforms import ToTensor

CONFIG = Dict(yaml.safe_load(open('./result/danet_segnet/config.yaml')))

data = PASCALVOC(
    CONFIG, 
    mode='train', 
    transform=transforms.Compose([ToTensor()])
)

data_loader = DataLoader(data, batch_size=1, shuffle=False)

mean = 0
std = 0
n = 0

for sample in data_loader:
    img = sample['image']
    img = img.view(len(img), 3, -1)
    mean += img.mean(2).sum(0)
    std += img.std(2).sum(0)
    n += len(img)

mean /= n
std /= n

print(mean)
print(std)
```
"""
