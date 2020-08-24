import torch
import random
import numpy as np

from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_building_train_dataset
from parsing.detector import RFJunctionDetector
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

def train_building():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=True,
                        )
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
    parser.add_argument("--checkpoint", help="checkpoint to be loaded")
    parser.add_argument("--checkpoint_dir", help="checkpoint dir")
    parser.add_argument("--resume_train", action="store_true", help="load checkpoint and resume training")
    parser.add_argument("--fp16", action="store_true", help="training in fp16 mode")
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RFJunctionDetector(cfg).to(device)
    train_dataset = build_building_train_dataset(cfg)
    if args.checkpoint and not args.resume_train:
        checkpoint = torch.load(args.checkpoint)
        filtered_weights = {
            k: v for k, v \
            in checkpoint['model_state_dict'].items() \
            if "mflow_conv_g6_b3_joint_drop_conv_new_2" not in k
        }
        model.backbone.load_state_dict(filtered_weights, strict=False)
    elif args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    params_res = []
    params_refine = []
    for k, v in model.named_parameters():
        if k.split('.')[1].startswith(("conv1", "bn_conv1", "res", "bn")):
            params_res.append(v)
        else:
            params_refine.append(v)
    optimizer = torch.optim.Adam([{"params": params_res, "lr": 0.000003}, {"params": params_refine, "lr": 0.00003}], betas=(0.9, 0.999), weight_decay=0.0005)

    if args.checkpoint and args.resume_train:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.train()

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
        if args.checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    total_loss, total_loss_main, total_loss_aux = 0.0, 0.0, 0.0
    gap = 100
    t = 0 if not args.resume_train else checkpoint["iter"]

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    for epoch in range(1000):
        for it, (images, target) in enumerate(train_dataset):
            images = images.to(device)
            target = target.to(device).long()
            
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(8, 16))
            # fig.add_subplot(1, 2, 1)
            # plt.imshow(images[0, ...].squeeze().permute(1, 2, 0))
            # fig.add_subplot(1, 2, 2)
            # plt.imshow(target[0, ...].squeeze())
            # plt.show()

            if args.fp16:
                with torch.cuda.amp.autocast():
                    y, contrastive_loss = model(images)
                    loss = loss_fn(y, target.half())
                    # contrastive_loss = sum(contrastive_loss)
            else:
                y, contrastive_loss = model(images)
                contrastive_loss = sum([torch.mean(c) for c in contrastive_loss if not isinstance(c, int)]) * 1e-3
                loss_main = loss_fn(y, target.squeeze(1))
                loss = loss_main + contrastive_loss

            total_loss += loss.item()
            total_loss_main += loss_main.item()
            total_loss_aux += contrastive_loss.item()
            if (t + 1) % gap == 0:
                print("iter: {}, loss_total: {}, loss_main: {}, loss_aux: {}".format(t + 1, total_loss/gap, total_loss_main/gap, total_loss_aux/gap))
                total_loss = 0.0
                total_loss_main = 0.0
                total_loss_aux = 0.0

            optimizer.zero_grad()
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
            if t % 10000 == 0 and t != 0:
                torch.save({
                    'iter': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scaler_state_dict': scaler.state_dict(),
                    'loss': loss,
                    },
                    "{}/checkpoint_finetune_iter_{}".format(args.checkpoint_dir, t)
                )
            t += 1


if __name__ == "__main__":
    train_building()
