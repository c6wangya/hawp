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
class HRJDataset(Dataset):
    def __init__(self, root, ann_file, transform = None, transform_target = None):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
        self.transform_target = transform_target

    def generate_target(self, junctions, sigma, height, width):
        tmp_size = sigma * 3
        target = np.zeros((height, width), dtype=np.float32)
        # loop over junctions
        for j in junctions:
            mu_x = j[1]
            mu_y = j[0]
            
            #check gaussian bounds
            ul = [
                max(int(mu_y - tmp_size), 0), 
                max(int(mu_x - tmp_size), 0)
            ]
            br = [
                min(int(mu_y + tmp_size + 1), height),
                min(int(mu_x + tmp_size + 1), width)
            ]

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            # Usable gaussian range
            g_ul = [y0 + ul[0] - int(mu_y), x0 + ul[1] - int(mu_x)]
            g_br = [y0 + br[0] - int(mu_y), x0 + br[1] - int(mu_x)]

            target[ul[1]: br[1], ul[0]: br[0]] = np.maximum(
                g[g_ul[1]: g_br[1], g_ul[0]: g_br[0]], 
                target[ul[1]: br[1], ul[0]: br[0]]
            )
        return target

    # for train
    def __getitem__(self, idx_):
        # print(idx_)
        idx = idx_%len(self.annotations)
        reminder = idx_//len(self.annotations)
        ann = copy.deepcopy(self.annotations[idx])
        ann['reminder'] = reminder
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        for key,_type in (['junctions',np.float32],
                          ['edges_positive',np.long],
                          ['edges_negative',np.long]):
            ann[key] = np.array(ann[key],dtype=_type)

        width = ann['width']
        height = ann['height']
        if reminder == 1:
            image = image[:,::-1,:]
            # image = F.hflip(image)
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
        elif reminder == 2:
            # image = F.vflip(image)
            image = image[::-1,:,:]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        elif reminder == 3:
            # image = F.vflip(F.hflip(image))
            image = image[::-1,::-1,:]
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        else:
            pass

        target = self.generate_target(
            ann['junctions'], 
            2, 
            height, 
            width
        )

        # # randomly crop
        # idx = [random.randint(0, 499 - 256), random.randint(0, 499 - 256)]
        # image = image[idx[0]:(idx[0] + 256), idx[1]:(idx[1] + 256), :]
        # target = target[idx[0]:(idx[0] + 256), idx[1]:(idx[1] + 256)]
 
        # fig = plt.figure(figsize=(8, 16))
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(image.astype('int'))
        # fig.add_subplot(1, 2, 2)
        # plt.imshow((target * 255).astype('int'), cmap='gray', vmin=0, vmax=255)
        # plt.show()

        if self.transform is not None:
            return self.transform(image), self.transform_target(target)
        return image, target

    def image(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        return image
    
    def get_image_name(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        return ann['filename']

    def __len__(self):
        return len(self.annotations)*4

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            default_collate([b[1] for b in batch]))
