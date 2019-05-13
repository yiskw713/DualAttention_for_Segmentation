import torch


def get_class_weight():
    """
        Class weight for CrossEntropy in Kinetics
        Class weight is calculated in the way described in:
            D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
            openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
        Class IDs can be obtained from `utils/class_label_map.py`
        if you calculate your dataset, please try this code:
    """

    class_weight = torch.tensor([
        0.0157, 1.6068, 3.7730, 1.2817, 1.8894, 1.8858, 0.6539, 0.8187, 0.4237,
        1.0000, 1.3883, 0.8461, 0.6585, 1.2528, 0.9905, 0.2385, 1.7129, 1.2691,
        0.7921, 0.7181, 1.2179
    ])

    return class_weight


"""
If you want to calculate the class weight by your own,
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

cnt_dict = {}
for i in range(21):
    cnt_dict[i] = 0
cnt_dict[255] = 0

for sample in data_loader:
    label = sample['label'].numpy()
    num, cnt = np.unique(label, return_counts=True)
    for n, c in zip(num, cnt):
        cnt_dict[n] += c

class_num = [cnt_dict[i] for i in range(21)]
class_num = torch.tensor(class_num)

total = class_num.sum().item()
frequency = class_num.float() / total
median = torch.median(frequency)
class_weight = median / frequency

```
"""
