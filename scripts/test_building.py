import torch
import random
import numpy as np

from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_building_test_dataset
from parsing.detector import RFJunctionDetector, WireframeDetector
from parsing.solver import make_lr_scheduler, make_optimizer
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
import os
import time
import datetime
import argparse
import logging
import pickle
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    a = a * mask
    weight = torch.ones(1, 1, 3, 3)
    a = a.unsqueeze(0)
    a = F.conv2d(a, weight, stride=1, padding=1)
    return a.squeeze(0)
    
def test_building():
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
    parser.add_argument("--hawp_checkpoint", help="pytorch checkpoint")
    parser.add_argument("--fp16", action="store_true", help="training in fp16 mode")
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFJunctionDetector(cfg).to(device)
    test_dataset = build_building_test_dataset(cfg)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.eval()

    # hawp_model = WireframeDetector(cfg, test_info='junc')
    # hawp_model = hawp_model.to(device)
    # logger = logging.getLogger("hawp.testing")
    # checkpointer = DetectronCheckpointer(cfg,
    #                                      hawp_model,
    #                                      save_dir=args.hawp_checkpoint,
    #                                      save_to_disk=True,
    #                                      logger=logger)
    # _ = checkpointer.load()
    # hawp_model = hawp_model.eval()

    for i, (images, target) in enumerate(tqdm(test_dataset)):
        with torch.no_grad():
            output = model(images.to(device))
            output = to_device(output,'cpu').squeeze(dim=0)
        
        target = target.squeeze(dim=0)

        if args.display:
            fig = plt.figure(figsize=(24, 8))
            # 01 origin
            fig.add_subplot(1, 3, 1)
            plt.imshow(((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().permute(1, 2, 0).int())
            plt.title('original image')
            # 02 prediction
            fig.add_subplot(1, 3, 2)
            plt.imshow((output[1, :, :] * 255 / output[1, :, :].max()).int(), cmap='gray', vmin=0, vmax=255)
            plt.title('prediction heatmap')
            # 03 gt
            fig.add_subplot(1, 3, 3)
            plt.imshow((target * 255 / target.max()).int().permute(1, 2, 0), cmap='gray', vmin=0, vmax=255)
            plt.title('ground truth')

            # plt.show()
            if not os.path.exists("./deform_attn_test"):
                os.makedirs("./deform_attn_test")
            save_dir = "./deform_attn_test/base_20k/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + "{}.png".format(test_dataset.dataset.filename(i)))
    
        # if args.display:
        #     fig = plt.figure(figsize=(32, 24))
        #     fig.add_subplot(3, 4, 1)
        #     plt.imshow(((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().permute(1, 2, 0).int())
        #     plt.title('original image')

        #     juncs = hawp_output.round().int().numpy()
        #     fig.add_subplot(3, 4, 2)
        #     hawp_image = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().permute(1, 2, 0).int().clone()
        #     plt.imshow(hawp_image)
        #     plt.plot(juncs[:, 0], juncs[:, 1], 'ro', markersize=2)
        #     plt.title('hawp (hourglass)')

            
        #     fig.add_subplot(3, 4, 3)
        #     image_copy = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().clone().permute(1, 2, 0)

        #     output_copy = output.clone()
            
        #     threshold = 0.002
        #     bool_map = (output_copy < threshold).int()
        #     image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        #     image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        #     image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map

        #     output_copy[output_copy < threshold] = 0
        #     output_copy[output_copy >= threshold] = 255
        #     # image_copy[:, :, 0] = output_copy

        #     image_copy[:, :, 0] = image_copy[:, :, 0] + output_copy
        #     # image_copy[:, :, 1] = image_copy[:, :, 1] + output_copy
        #     # image_copy[:, :, 2] = image_copy[:, :, 2] + output_copy

        #     # image_copy[1, :, :] = image_copy[1, :, :] + output_copy * (255 / output_copy.max())
        #     plt.imshow(image_copy.int())
        #     plt.title('refinenet --threshold 0.002')

        #     fig.add_subplot(3, 4, 4)
        #     image_copy = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().clone().permute(1, 2, 0)

        #     output_copy = output.clone()
            
        #     threshold = 0.005
        #     bool_map = (output_copy < threshold).int()
        #     image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        #     image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        #     image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map

        #     output_copy[output_copy < threshold] = 0
        #     output_copy[output_copy >= threshold] = 255
        #     # image_copy[:, :, 0] = output_copy

        #     image_copy[:, :, 0] = image_copy[:, :, 0] + output_copy
        #     # image_copy[:, :, 1] = image_copy[:, :, 1] + output_copy
        #     # image_copy[:, :, 2] = image_copy[:, :, 2] + output_copy

        #     # image_copy[1, :, :] = image_copy[1, :, :] + output_copy * (255 / output_copy.max())
        #     plt.imshow(image_copy.int())
        #     plt.title('refinenet --threshold 0.005')

        #     fig.add_subplot(3, 4, 5)
        #     image_copy = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().clone().permute(1, 2, 0)

        #     output_copy = output.clone()
            
        #     threshold = 0.02
        #     bool_map = (output_copy < threshold).int()
        #     image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        #     image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        #     image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map

        #     output_copy[output_copy < threshold] = 0
        #     output_copy[output_copy >= threshold] = 255
        #     # image_copy[:, :, 0] = output_copy

        #     image_copy[:, :, 0] = image_copy[:, :, 0] + output_copy
        #     # image_copy[:, :, 1] = image_copy[:, :, 1] + output_copy
        #     # image_copy[:, :, 2] = image_copy[:, :, 2] + output_copy

        #     # image_copy[1, :, :] = image_copy[1, :, :] + output_copy * (255 / output_copy.max())
        #     plt.imshow(image_copy.int())
        #     plt.title('refinenet --threshold 0.02')

        #     fig.add_subplot(3, 4, 6)
        #     image_copy = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().clone().permute(1, 2, 0)

        #     output_copy = output.clone()
            
        #     threshold = 0.03
        #     bool_map = (output_copy < threshold).int()
        #     image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        #     image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        #     image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map

        #     output_copy[output_copy < threshold] = 0
        #     output_copy[output_copy >= threshold] = 255
        #     # image_copy[:, :, 0] = output_copy

        #     image_copy[:, :, 0] = image_copy[:, :, 0] + output_copy
        #     # image_copy[:, :, 1] = image_copy[:, :, 1] + output_copy
        #     # image_copy[:, :, 2] = image_copy[:, :, 2] + output_copy

        #     # image_copy[1, :, :] = image_copy[1, :, :] + output_copy * (255 / output_copy.max())
        #     plt.imshow(image_copy.int())
        #     plt.title('refinenet --threshold 0.03')
            
        #     fig.add_subplot(3, 4, 7)
        #     image_copy = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().clone().permute(1, 2, 0)

        #     output_copy = output.clone()
            
        #     threshold = 0.04
        #     bool_map = (output_copy < threshold).int()
        #     image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        #     image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        #     image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map

        #     output_copy[output_copy < threshold] = 0
        #     output_copy[output_copy >= threshold] = 255
        #     # image_copy[:, :, 0] = output_copy

        #     image_copy[:, :, 0] = image_copy[:, :, 0] + output_copy
        #     # image_copy[:, :, 1] = image_copy[:, :, 1] + output_copy
        #     # image_copy[:, :, 2] = image_copy[:, :, 2] + output_copy

        #     # image_copy[1, :, :] = image_copy[1, :, :] + output_copy * (255 / output_copy.max())
        #     plt.imshow(image_copy.int())
        #     plt.title('refinenet --threshold 0.04')

        #     fig.add_subplot(3, 4, 8)
        #     image_copy = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().clone().permute(1, 2, 0)

        #     output_copy = output.clone()
            
        #     threshold = 0.05
        #     bool_map = (output_copy < threshold).int()
        #     image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        #     image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        #     image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map

        #     output_copy[output_copy < threshold] = 0
        #     output_copy[output_copy >= threshold] = 255
        #     # image_copy[:, :, 0] = output_copy

        #     image_copy[:, :, 0] = image_copy[:, :, 0] + output_copy
        #     # image_copy[:, :, 1] = image_copy[:, :, 1] + output_copy
        #     # image_copy[:, :, 2] = image_copy[:, :, 2] + output_copy

        #     # image_copy[1, :, :] = image_copy[1, :, :] + output_copy * (255 / output_copy.max())
        #     plt.imshow(image_copy.int())
        #     plt.title('refinenet --threshold 0.05')

        #     fig.add_subplot(3, 4, 9)
        #     plt.imshow((raw_output * 255 / raw_output.max()).int().permute(1, 2, 0))
        #     plt.title('refinenet output heatmap')
        #     fig.add_subplot(3, 4, 10)
        #     plt.imshow((output * 255 / output.max()).int().permute(1, 2, 0))
        #     plt.title('refinenet NMS output heatmap')
        #     fig.add_subplot(3, 4, 11)
        #     plt.imshow((target * 255).int().permute(1, 2, 0))
        #     plt.title('GT heatmap')

        #     fig.add_subplot(3, 4, 12)
        #     image_copy = ((images - images.min()) * (255 / (images.max() - images.min()))).squeeze().clone().permute(1, 2, 0)

        #     bool_map = (target < 0.4).int()
        #     image_copy[:, :, 0] = image_copy[:, :, 0] * bool_map
        #     image_copy[:, :, 1] = image_copy[:, :, 1] * bool_map
        #     image_copy[:, :, 2] = image_copy[:, :, 2] * bool_map
        #     target[target < 0.4] = 0
        #     image_copy[:, :, 0] = image_copy[:, :, 0] + (target * 255).int()
        #     plt.imshow(image_copy.int())
        #     plt.title('GT')

        #     #plt.show()
        #     save_dir = "./refinenet_test/comparison_v2_360k/"
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     plt.savefig(save_dir + "{}.png".format(i))

if __name__ == "__main__":
    test_building()
