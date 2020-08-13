import torch
import parsing
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset.build import build_transform
from parsing.detector import WireframeDetector
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
import torch.nn.functional as F
from tqdm import tqdm
import json
from matplotlib.patches import Circle
import pathlib
import numpy as np

parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--config-file",
                    metavar="FILE",
                    help="path to config file",
                    type=str,
                    required=True,
                    )

parser.add_argument("--img",type=str,required=True,
                    help="image path")                    

parser.add_argument("--info",type=str, required=True, 
                    help="which info is required (junc or ls)")

parser.add_argument("--checkpoint",type=str, required=True, 
                    help="checkpoint directory")

args = parser.parse_args()

def enhance(a):
    weight = torch.ones(1, 1, 3, 3)
    a = a.unsqueeze(0).unsqueeze(1).float()
    a = F.conv2d(a, weight, stride=1, padding=1)
    return a.squeeze()

def test(cfg, impath):
    logger = logging.getLogger("hawp.testing")
    device = cfg.MODEL.DEVICE
    model = WireframeDetector(cfg, test_info=args.info)
    model = model.to(device)

    transform = build_transform(cfg)
    image = io.imread(impath)[:,:,:3]
    image_tensor = transform(image.astype(float))[None].to(device)
    meta = {
        'filename': impath,
        'height': image.shape[0],
        'width': image.shape[1],
    }
    
    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=args.checkpoint,
                                         save_to_disk=True,
                                         logger=logger)
    _ = checkpointer.load()
    model = model.eval()

    with torch.no_grad():
        output, _ = model(image_tensor,[meta])
        output = to_device(output,'cpu')

    if args.info == "junc":
        juncs = output.numpy()
        plt.figure(figsize=(6,6))    
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        #         hspace = 0, wspace = 0)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for xx,yy in zip(juncs[:, 0], juncs[:, 1]):
            circ = Circle((int(xx), int(yy)), 1, color='r', fill=False)
            ax.add_patch(circ)
        # ax.plot(juncs[:, 0], juncs[:, 1], 'o', color='y', fill=False)
        pathlib.Path("./test_results").mkdir(parents=True, exist_ok=True)
        plt.savefig("./test_results/test_{}.png".format(args.info))
        plt.show()
        plt.close('all')
    if args.info == "ls":
        fig = plt.figure(figsize=(24, 24))
        fig.add_subplot(1, 1, 1)
        dis0_points = output.numpy()
        image_copy = torch.from_numpy(image.copy())
        points_map = np.zeros((image.shape[0], image.shape[1]))
        for x, y in zip(dis0_points[:, 1], dis0_points[:, 0]):
            points_map[round(x), round(y)] = 255
        points_map = torch.from_numpy(points_map)
        points_map = enhance(points_map)
        bool_map = (points_map == 0).int()
        image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map
        image_copy[:, :, 0] = image_copy[:, :, 0] + points_map.to(torch.uint8)

        plt.imshow(image_copy)
        plt.show()

        lines = output.numpy()
        plt.figure(figsize=(6,6))    
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
        plt.imshow(image)
        plt.plot([lines[:, 0],lines[:, 2]],
                            [lines[:, 1],lines[:, 3]], color='b')
        # plt.plot(lines[:, 0],lines[:, 1],'c')
        # plt.plot(lines[:, 2],lines[:, 3],'c')
        plt.axis('off')
        pathlib.Path("./test_results").mkdir(parents=True, exist_ok=True)
        plt.savefig("./test_results/test_{}.png".format(args.info))
        plt.show()
        plt.close('all')
    
if __name__ == "__main__":

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('hawp', output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))


    test(cfg,args.img)

