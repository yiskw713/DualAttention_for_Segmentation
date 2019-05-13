import os

from PIL import Image
from torch.utils.data import Dataset


class PASCALVOC(Dataset):
    """
    PASCAL VOC Segmentation dataset
    """

    def __init__(self, config, mode="train", transform=None):
        super().__init__()
        self.config = config
        self.mode = mode    # ['train', 'trainval', 'val', 'test']
        self.transform = transform

        # set file pathes
        root = os.path.join(
            self.config.dataset_dir, "VOC{}".format(self.config.year))
        self.image_dir = os.path.join(root, "JPEGImages")
        self.label_dir = os.path.join(root, "SegmentationClass")

        if self.mode in ["train", "trainval", "val", "test"]:
            txt = os.path.join(
                root, "ImageSets/Segmentation", self.mode + ".txt")
            files = list(open(txt, "r"))
            files = [i.rstrip() for i in files]
            self.files = files
        else:
            raise ValueError("Invalid split name: {}".format(self.mode))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # paths to an image and a label
        image_id = self.files[idx]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        label_path = os.path.join(self.label_dir, image_id + ".png")

        # Load an image and a label
        image = Image.open(image_path)
        label = Image.open(label_path)

        sample = {
            "id": image_id,
            "image": image,
            "label": label,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
