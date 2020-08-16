import torch
from torch.utils.data import Dataset

import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy
import glob

class BuildingTestDataset(Dataset):
    def __init__(self, root, transform = None, transform_target = None):
        self.root = root
        self.transform = transform
        self.transform_target = transform_target
        image_files = glob.glob(root + '/test_images/*')
        self.images = {fpth.split('/')[-1][:-4]: fpth for fpth in image_files}
        label_files = glob.glob(root + '/test_labels/*')
        self.labels = {fpth.split('/')[-1][:-4]: fpth for fpth in label_files}

    # for train
    def __getitem__(self, idx_):
        # print(idx_)
        idx = idx_ % len(self.images.keys())
        file_name = list(self.images.keys())[idx]
        image = io.imread(self.images[file_name]).astype(float)[:,:,:3]
        target = io.imread(self.labels[file_name], as_gray=True).round().astype(float)
        if self.transform is not None:
            return self.transform(image), self.transform_target(target)
        return image, target

    def filename(self, idx_):
        return list(self.images.keys())[idx_]

    def __len__(self):
        return len(self.images.keys())

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            default_collate([b[1] for b in batch]))
