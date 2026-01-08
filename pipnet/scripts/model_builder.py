#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:19:24 2023

@author: lisadesanti
Updated for Multimodal PIPNet
"""

import sys
import os
from copy import deepcopy
import torch
import torchvision.models as models
import torch.nn as nn

from utils import get_model_layers
from utils import set_device
from utils import get_optimizer_nn
from pipnet import get_network, PIPNet
# from convnext_features import convnext_tiny_3d # Behåll om du använder convnext, annars kan den tas bort


def load_trained_pipnet(args):
    
    models_folder = os.path.join(args.model_path, "binary", args.model_name)

    # Bygg sökvägen (kan behöva justeras om dina sparade modeller har .pth ändelse)
    model_name = f"best_pipnet_fold{args.current_fold}"
    model_path = os.path.join(models_folder, args.net, model_name)

    device, device_ids = set_device(args)
     
    # --- UPDATE: Create Multimodal 3D-PIPNet ---
    # get_network returnerar nu 5 värden:
    # backbones (ModuleDict), add_ons (ModuleDict), pool, classification, num_protos
    backbones, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(args.out_shape, args)
    
    # --- UPDATE: Instantiate with correct arguments ---
    net = PIPNet(
        num_classes = args.out_shape,
        backbones = backbones,            # Var 'feature_net' förut
        add_on_layers = add_on_layers,
        pool_layer = pool_layer,
        classification_layer = classification_layer
        )
    
    net = net.to(device = device)
    net = nn.DataParallel(net, device_ids = device_ids)  
    
    # Load trained network
    print(f"[INFO] Loading model from {model_path}...")
    
    # Hantera om filen har .pth suffix eller inte
    final_path = model_path
    if not os.path.exists(final_path):
        if os.path.exists(final_path + ".pth"):
            final_path = final_path + ".pth"
        else:
            print(f"[ERROR] Model file not found at {model_path} (or with .pth extension)")
            # Vi låter den krascha nedan om filen saknas, eller så kan du returnera None
    
    # Ladda med map_location för att undvika device-mismatch
    checkpoint = torch.load(final_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    net.to(device)
    net.eval()
    
    return net