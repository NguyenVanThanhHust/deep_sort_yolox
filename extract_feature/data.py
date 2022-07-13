import argparse
import os
import time
from os.path import join, isdir, isfile

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset

class Market1501(Dataset):
    def __init__(self, data_dir, split="train", transform=None) -> None:
        super().__init__()
        self.img_dir = join(data_dir, "bounding_box_" + split)
        self.img_list = next(os.walk(self.img_dir))[2]
        tmp_list = [s.split("_")[0] for s in self.img_list]
        tmp_set = set(tmp_list)
        self.ids = list(tmp_set)
        self.classes = self.ids
        self.transform = transform

    def __len__(self,):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = join(self.img_dir, self.img_list[index])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        img_id = self.img_list[index].split("_")[0]
        label = np.array(self.ids.index(img_id))
        return img, torch.from_numpy(label)


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128,64),padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# data = Market1501(data_dir="Market-1501-v15.09.15", transform=transform_train)

# for i in range(10):
#     item = data.__getitem__(i)
#     img, label = item
#     print(img.shape, label)