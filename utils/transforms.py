import numpy as np
import random
import torch

from torchvision import transforms

from utils.mean import get_mean


class RandomCrop(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        r, g, b = get_mean(norm_value=1)
        self.mean = (int(r), int(g), int(b))

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        W, H = image.size
        print('original', image.size)
        if H < self.config.height:
            delta_height = self.config.height - H
            image = transforms.functional.pad(
                image, (0, delta_height - delta_height//2), fill=self.mean)
            label = transforms.functional.pad(
                label, (0, delta_height - delta_height//2), fill=255)

        if W < self.config.width:
            delta_width = self.config.width - W
            image = transforms.functional.pad(
                image, (delta_width - delta_width // 2, 0), fill=self.mean)
            label = transforms.functional.pad(
                label, (delta_width - delta_width // 2, 0), fill=255)

        print('padded', image.size)
        i, j, h, w = transforms.RandomCrop.get_params(
            image, (self.config.crop_height, self.config.crop_width))
        image = transforms.functional.crop(image, i, j, h, w)
        label = transforms.functional.crop(label, i, j, h, w)

        sample['image'] = image
        sample['label'] = label
        return sample


class Resize(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = transforms.functional.resize(
            image, (self.config.height, self.config.width))
        label = transforms.functional.resize(
            label, (self.config.height, self.config.width), interpolation=1)

        sample['image'] = image
        sample['label'] = label
        return sample


class RandomFlip(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            label = sample['label']

            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

            sample['image'] = image
            sample['label'] = label

            return sample

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        sample['image'] = transforms.functional.to_tensor(image).float()
        label = np.asarray(label, dtype=np.int64)
        sample['label'] = torch.from_numpy(label)
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image = transforms.functional.normalize(image, self.mean, self.std)
        sample['image'] = image
        return sample
