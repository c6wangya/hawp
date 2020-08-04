import torch
import parsing
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset.build import build_transform
from parsing.detector import RFJunctionDetector
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
from skimage import io
import os
import os.path as osp
import time
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from matplotlib.patches import Circle
import pathlib

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=True,
                        )
    parser.add_argument("--display",
                    default=False,
                    action='store_true')
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
    parser.add_argument("--checkpoint", help="pytorch checkpoint")
    parser.add_argument("--ann_path", help="annotation path")
    parser.add_argument("--im_path", help="image path")
    parser.add_argument("--fp16", action="store_true", help="training in fp16 mode")
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFJunctionDetector(cfg).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    transform = build_transform(cfg)
    image = io.imread(args.im_path)[:,:,:3]
    image_tensor = transform(image.astype(float))[None].to(device)
    meta = {
        'filename': args.im_path,
        'height': image.shape[0],
        'width': image.shape[1],
    }
    
    model = model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        output = to_device(output,'cpu').squeeze(dim=0)

    # target = target.squeeze(dim=0)
        
    if args.display:
        threshold = 0.1
        
        fig = plt.figure(figsize=(8, 24))
        fig.add_subplot(1, 4, 1)

        plt.imshow(((image_tensor - image_tensor.min()) * (255 / (image_tensor.max() - image_tensor.min()))).squeeze().permute(1, 2, 0).int())
        
        fig.add_subplot(1, 4, 2)
        image_copy = ((image_tensor - image_tensor.min()) * (255 / (image_tensor.max() - image_tensor.min()))).squeeze()
        # image_copy = image_copy.numpy()
        # heat_idx = (output > threshold).nonzero().numpy()
        bool_map = (output < threshold).int()
        image_copy[0, ...] = image_copy[0, ...] * bool_map
        image_copy[1, ...] = image_copy[1, ...] * bool_map
        output[output < threshold] = 0
        image_copy[0, ...] = image_copy[0, ...] + output * (255 / output.max())
        image_copy[1, ...] = image_copy[1, ...] + output * (255 / output.max())
        # image_copy[heat_idx] = output[heat_idx] * 255
        plt.imshow(image_copy.permute(1, 2, 0).int())
        fig.add_subplot(1, 4, 3)
        plt.imshow((output * 255).int().permute(1, 2, 0))
        fig.add_subplot(1, 4, 4)
        plt.imshow((target * 255).int().permute(1, 2, 0))
        plt.show()
        plt.savefig("./test_fig.png")
    
if __name__ == "__main__":
    predict()