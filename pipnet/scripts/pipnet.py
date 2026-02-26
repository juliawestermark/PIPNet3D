#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:00:52 2023

@author: lisadesanti
"""

import argparse
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from videoresnet_features import video_resnet18_features
from convnext_features import convnext_tiny_3d_features


class PIPNet(nn.Module):
    
    def __init__(self,
                 num_classes: int,
                 backbones: nn.ModuleDict,      
                 add_on_layers: nn.ModuleDict,  
                 pool_layer: nn.Module,
                 classification_layer: nn.Module,
                 default_threshold: float = 0.1 
                 ):
        
        super().__init__()
        self._num_classes = num_classes
        self._backbones = backbones
        self._add_ons = add_on_layers
        self.modalities = list(self._backbones.keys())
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier
        self.default_threshold = default_threshold

    def forward(self, xs: dict, masks: dict = None, inference=False, threshold=None):
        """
        threshold: Can be:
                   1. None  -> Uses self.default_threshold (same for all)
                   2. float -> e.g., 0.1 (same for all)
                   3. dict  -> e.g., {'mri': 0.3, 'amy': 0.05} (modality-specific)
        """
        
        proto_features_dict = {}
        pooled_list = []

        # Prepare threshold logic
        current_thresholds = {}
        if threshold is None:
            # Case 1: Use default (float)
            for m in self.modalities: current_thresholds[m] = self.default_threshold
        elif isinstance(threshold, float) or isinstance(threshold, int):
            # Case 2: A float was provided (same for all)
            for m in self.modalities: current_thresholds[m] = float(threshold)
        elif isinstance(threshold, dict):
            # Case 3: A dictionary was provided (modality-specific)
            current_thresholds = threshold
            # Fill with default if any modality is missing in the dict
            for m in self.modalities:
                if m not in current_thresholds:
                    current_thresholds[m] = self.default_threshold

        for modality in self.modalities:
            x = xs[modality]
            
            # 1. Backbone
            features = self._backbones[modality](x)
            
            # 2. Add-on (Prototypes)
            proto_features = self._add_ons[modality](features)
            proto_features_dict[modality] = proto_features
            
            # 3. Pooling -> (bs, num_prototypes)
            pooled = self._pool(proto_features) 
            
            # 4. MASKING (Handle missing data)
            if masks is not None and modality in masks:
                mask = masks[modality].to(pooled.device)
                pooled = pooled * mask
            
            if inference:
                # Get threshold for this specific modality
                thr = current_thresholds[modality]
                
                # Zero out everything below the threshold
                pooled = torch.where(pooled < thr, 0., pooled)

            pooled_list.append(pooled)

        # 5. Fusion (List already consists of thresholded vectors if inference=True)
        pooled_combined = torch.cat(pooled_list, dim=1) 

        # Classification
        out = self._classification(pooled_combined)
        
        # Return
        if inference:
            return proto_features_dict, pooled_combined, out
        else:
            return proto_features_dict, pooled_combined, out
        
        
base_architecture_to_features = {
    'resnet3D_18_kin400': video_resnet18_features,
    'convnext3D_tiny': convnext_tiny_3d_features,
    }

# adapted from 
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    
    """
    Applies a linear transformation to the incoming data with non-negative weights` """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 device = None, 
                 dtype = None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(
            torch.ones((1,), requires_grad = True))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)

def _get_backbone_channels(args, features):
    """Helper to find output channels of the backbone"""
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()
        
    if features_name.startswith('VIDEO') or features_name.startswith('RES') or features_name.startswith('CONV'):
        # Find the last Conv3d layer
        return [i for i in features.modules() if isinstance(i, nn.Conv3d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')


def _create_add_on_layer(in_channels, num_prototypes):
    """Helper to create the prototype (add-on) layer"""
    if num_prototypes == 0:
        # If num_features is 0, the number of channels from the backbone is used as the number of prototypes
        # This is standard PIPNet behavior
        return nn.Sequential(nn.Softmax(dim=1),), in_channels
    else:
        print(f"Number of prototypes set from {in_channels} to {num_prototypes}. 1x1x1 conv layer added.", flush=True)
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=num_prototypes, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.Softmax(dim=1),
        ), num_prototypes
    
def get_network(num_classes: int, args: argparse.Namespace): 
    
    modalities = args.modalities

    backbones = nn.ModuleDict()
    add_ons = nn.ModuleDict()
    total_prototypes = 0
    prototypes_per_modality = {}

    print(f"Building Multi-Modal PIPNet for: {modalities}", flush=True)

    channels = 1
    
    for mod in modalities:
        
        print(f"  initializing backbone for {mod} (channels={channels})...", flush=True)
        
        # 1. Create Backbone
        backbone = base_architecture_to_features[args.net](
            pretrained = not args.disable_pretrained
        )
        
        # 2. Find output channels
        backbone_out_channels = _get_backbone_channels(args, backbone) 

        # 3. Create Add-on layer
        add_on, n_protos = _create_add_on_layer(backbone_out_channels, args.num_features)
        
        # 4. Add to ModuleDicts
        backbones[mod] = backbone
        add_ons[mod] = add_on
        
        total_prototypes += n_protos
        prototypes_per_modality[mod] = n_protos

    # 5. Pooling (shared)
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool3d(output_size=(1,1,1)), 
        nn.Flatten()
    )

    # 6. Classification
    print(f"Total prototypes: {total_prototypes} {prototypes_per_modality}", flush=True)
    
    if args.bias:
        classification_layer = NonNegLinear(total_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(total_prototypes, num_classes, bias=False)
        
    # Return ModuleDicts instead of individual layers
    return backbones, add_ons, pool_layer, classification_layer, total_prototypes