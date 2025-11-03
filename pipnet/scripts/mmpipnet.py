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
from typing import List


class MMPIPNet(nn.Module):
    
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int, # TODO: Remove this or use a list?
                 feature_nets: List[nn.Module],
                 args: argparse.Namespace,
                 add_on_layers_list: List[nn.Module],
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        
        super().__init__()
        assert num_classes > 0
        assert len(feature_nets) == len(add_on_layers_list)
        self._num_modalities = len(feature_nets)
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._nets = nn.ModuleList(feature_nets)
        self._add_ons = nn.ModuleList(add_on_layers_list)    # Softmax over ps
        self._pool = pool_layer         # AdaptiveMaxPooling3D -> Flattening
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier
        

    def forward(self, xs_list, inference=False):
        
        # TODO: Update for non-equal number of prototypes per modality
        if len(xs_list) != len(self._nets):
            raise ValueError(f"Expected {len(self._nets)} inputs, but got {len(xs_list)}.")
        proto_features_list = []
        pooled_list = []
        modality_index_list = []

        for i in range(len(xs_list)):
            xs = xs_list[i]
            features = self._nets[i](xs) 
            proto_features = self._add_ons[i](features) # (bs,ps,d,h,w)
            pooled = self._pool(proto_features)     # (bs,ps,1,1,1) -> (bs,ps): prototype presence in xs
            proto_features_list.append(proto_features)
            pooled_list.append(pooled)
            mod_indices = torch.full(
                (pooled.shape[1],), i, dtype=torch.long, device=pooled.device
            )
            modality_index_list.append(mod_indices)
        
        # proto_features = torch.cat(proto_features_list, dim=1) # (bs, total_ps, d, h, w)
        pooled = torch.cat(pooled_list, dim=1)                 # (bs, total_ps)
        modality_indices = torch.cat(modality_index_list, dim=0)
        
        if inference:
            # during inference, ignore all prototypes that have 0.1 similarity  or lower
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled) # (bs,ps)
            out = self._classification(clamped_pooled) # (bs,num_classes)
            return proto_features_list, clamped_pooled, modality_indices, out
        
        else:
            out = self._classification(pooled) # (bs*2,num_classes) 
            return proto_features_list, pooled, modality_indices, out
        
        
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

    
def get_network(num_classes: int, args: argparse.Namespace): 
    

    features = base_architecture_to_features[args.net](pretrained = not args.disable_pretrained)
    
    features_name = str(features).upper()
    
    if 'next' in args.net:
        features_name = str(args.net).upper()
        
    if features_name.startswith('VIDEO') or features_name.startswith('RES') or features_name.startswith('CONV'):
        first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv3d)][-1].out_channels

    else:
        raise Exception('other base architecture NOT implemented')
    
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(nn.Softmax(dim=1),)  # softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
              
    else:
        num_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes, ". Extra 1x1x1 conv layer added. Not recommended.", flush=True)
        
        add_on_layers = nn.Sequential(
            nn.Conv3d(in_channels = first_add_on_layer_in_channels, out_channels = num_prototypes, kernel_size = 1, stride = 1, padding = 0, bias = True), 
            nn.Softmax(dim=1),)  # softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
             
    # pool_layer = nn.Sequential(
    #     nn.AdaptiveMaxPool3d(output_size=(1,1,1)), # dim: (bs,ps,1,1,1) 
    #     nn.Flatten())                              # dim: (bs,ps)
         
    # if args.bias:
    #     classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)

    # else:
    #     classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return features, add_on_layers, num_prototypes
    # return features, add_on_layers, pool_layer, classification_layer, num_prototypes


def get_mmpipnet_network(num_classes: int, args: argparse.Namespace, num_modalities: int): 
    
    feature_nets = []
    add_on_layers_list = []
    total_num_prototypes = 0
    
    for i in range(num_modalities):
        features, add_on_layers, num_prototypes = get_network(num_classes, args)
        feature_nets.append(features)
        add_on_layers_list.append(add_on_layers)
        total_num_prototypes += num_prototypes
    
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool3d(output_size=(1,1,1)), # dim: (bs,ps,1,1,1) 
        nn.Flatten())                              # dim: (bs,ps)
         
    if args.bias:
        classification_layer = NonNegLinear(total_num_prototypes, num_classes, bias=True)

    else:
        classification_layer = NonNegLinear(total_num_prototypes, num_classes, bias=False)
        
    return feature_nets, add_on_layers_list, pool_layer, classification_layer, total_num_prototypes



