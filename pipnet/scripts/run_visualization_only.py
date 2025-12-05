#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to run visualization without training
"""

import os
import sys
import torch
import torch.nn as nn
import random
import numpy as np

from utils import set_device, get_args
from make_dataset import get_dataloaders
from pipnet import get_network, PIPNet
from vis_pipnet import visualize_topk


# Configuration
backbone_dic = {1:"resnet3D_18_kin400", 2:"convnext3D_tiny"}
current_fold = 1
net = backbone_dic[1]
task_performed = "train_pipnet_alzheimer_mri"

args = get_args(current_fold, net, task_performed)

# Set the checkpoint path - using the best trained model
args.state_dict_dir_net = '/home/maia-user/PIPNet3D/results/train_pipnet_alzheimer_mri/resnet3D_18_kin400/fold_1/checkpoints/best_pipnet_fold1'

# Alternative checkpoints you can use:
# args.state_dict_dir_net = '/home/maia-user/PIPNet3D/results/train_pipnet_alzheimer_mri/resnet3D_18_kin400/fold_1/checkpoints/net_trained_last'  # Last trained
# args.state_dict_dir_net = '/home/maia-user/PIPNet3D/pipnet/models/binary/resnet3D_18_kin400/best_pipnet_fold1'  # From models directory

# Set seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Get dataloaders
print("Loading dataloaders...", flush=True)
dataloaders = get_dataloaders(args)
projectloader = dataloaders[4]  # Only need projectloader for visualization

# Set device
device, device_ids = set_device(args)
print(f"Using device: {device}", flush=True)

# Create network architecture
print("Creating network architecture...", flush=True)
network_layers = get_network(args.out_shape, args)
feature_net = network_layers[0]
add_on_layers = network_layers[1]
pool_layer = network_layers[2]
classification_layer = network_layers[3]
num_prototypes = network_layers[4]

net = PIPNet(
    num_classes=args.out_shape,
    num_prototypes=num_prototypes,
    feature_net=feature_net,
    args=args,
    add_on_layers=add_on_layers,
    pool_layer=pool_layer,
    classification_layer=classification_layer
)

net = net.to(device=device)
net = nn.DataParallel(net, device_ids=device_ids)

# Load pretrained model
if args.state_dict_dir_net != '':
    print(f"Loading pretrained model from: {args.state_dict_dir_net}", flush=True)
    checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("Pretrained model loaded successfully", flush=True)
else:
    print("ERROR: No pretrained model specified!", flush=True)
    print("Please set args.state_dict_dir_net to point to your trained model checkpoint", flush=True)
    sys.exit(1)

# Get latent dimensions with a forward pass
print("Getting latent dimensions...", flush=True)
with torch.no_grad():
    xs1, _ = next(iter(projectloader))
    xs1 = xs1.to(device)
    proto_features, _, _ = net(xs1)
    args.wshape = proto_features.shape[-1]
    args.hshape = proto_features.shape[-2]
    args.dshape = proto_features.shape[-3]
    print(f"Latent output shape: {proto_features.shape}", flush=True)

# Run visualization
print("\n" + "="*60, flush=True)
print("Running visualization...", flush=True)
print("="*60 + "\n", flush=True)

net.eval()
topks, img_prototype, proto_coord = visualize_topk(
    net,
    projectloader,
    args.num_classes,
    device,
    'visualised_prototypes_topk',
    args,
    save=True,  # Save visualization arrays as .npy files
    k=10,       # Top-10 most activated images per prototype
    plot=True   # Generate and save plots (set to False for faster execution)
)

print("\n" + "="*60, flush=True)
print("Visualization complete!", flush=True)
print(f"Results saved to: {os.path.join(args.log_dir, 'visualised_prototypes_topk')}", flush=True)
print("="*60 + "\n", flush=True)
