#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:19:24 2023

@author: lisadesanti
"""

import sys
import os
from copy import deepcopy
import torch
import torchvision.models as models
import torch.nn as nn

from utils import get_model_layers
from utils import set_device
from utils import get_optimizer_mm_nn
from pipnet import get_network, PIPNet
from mmpipnet import MMPIPNet, get_mmpipnet_network
from convnext_features import convnext_tiny_3d

def load_trained_mmpipnet(args):
    #models_folder = "/home/maia-user/PIPNet3D/pipnet/models/binary"
    models_folder = "/proj/berzbiomedicalimagingkth/users/x_julwe/PIPNet3D/pipnet/models/binary"
    model_path = os.path.join(models_folder, args.net, "best_mmpipnet_fold%s"%str(args.current_fold))

    device, device_ids = set_device(args)
     
    # Create MM 3D-PIPNet
    num_modalities = 1
    feature_nets, add_on_layers_list, pool_layer, classification_layer, total_num_prototypes = get_mmpipnet_network(args.out_shape, args, num_modalities)
    
    net = MMPIPNet(
        num_classes = args.out_shape,
        num_prototypes = total_num_prototypes,
        feature_nets = feature_nets,
        args = args,
        add_on_layers_list = add_on_layers_list,
        pool_layer = pool_layer,
        classification_layer = classification_layer
        )
    
    net = net.to(device = device)
    net = nn.DataParallel(net, device_ids = device_ids)  
    
    # Load trained network
    net_trained_last = torch.load(model_path)
    net.load_state_dict(net_trained_last['model_state_dict'], strict=True)
    net.to(device)
    net.eval()
    
    return net

