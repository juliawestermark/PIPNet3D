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
                 backbones: nn.ModuleDict,      # Dict: {'mri': net, 'pet': net}
                 add_on_layers: nn.ModuleDict,  # Dict: {'mri': layer, 'pet': layer}
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        
        super().__init__()
        self._num_classes = num_classes
        
        # Viktigt: nn.ModuleDict registrerar sub-modulerna korrekt i PyTorch
        self._backbones = backbones
        self._add_ons = add_on_layers
        
        # Vi sparar nycklarna (t.ex. ['mri', 'pet']) för att garantera ordningen 
        # när vi slår ihop vektorerna (concat).
        self.modalities = list(self._backbones.keys())
        
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs: dict, masks: dict = None, inference=False):
        """
        xs: Dict {'mri': tensor, 'pet': tensor}
        masks: Dict {'mri': tensor(bs, 1), 'pet': tensor(bs, 1)} (1=present, 0=missing)
        """
        
        proto_features_dict = {}
        pooled_list = []

        for modality in self.modalities:
            x = xs[modality]
            
            # 1. Backbone features
            # (Även noll-bild går igenom här, det är onödig beräkning men enklast kodmässigt)
            features = self._backbones[modality](x)
            
            # 2. Add-on (Prototyper)
            proto_features = self._add_ons[modality](features)
            proto_features_dict[modality] = proto_features
            
            # 3. Pooling -> (bs, num_prototypes)
            pooled = self._pool(proto_features) 
            
            # 4. MASKING (Här sker magin)
            if masks is not None and modality in masks:
                # masks[modality] har shape (bs, 1). pooled har (bs, ps).
                # Broadcasting ser till att alla prototyper nollas för det samplet.
                mask = masks[modality].to(pooled.device)
                pooled = pooled * mask
            
            pooled_list.append(pooled)

        # 5. Fusion
        pooled_combined = torch.cat(pooled_list, dim=1) 

        if inference:
            clamped_pooled = torch.where(pooled_combined < 0.1, 0., pooled_combined)
            out = self._classification(clamped_pooled)
            return proto_features_dict, clamped_pooled, out
        
        else:
            out = self._classification(pooled_combined)
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
        # Hitta sista Conv3d lagret
        return [i for i in features.modules() if isinstance(i, nn.Conv3d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')


def _create_add_on_layer(in_channels, num_prototypes):
    """Helper to create the prototype (add-on) layer"""
    if num_prototypes == 0:
        # Om num_features är 0 används antalet kanaler från backbone som antal prototyper
        # Detta är standard PIPNet beteende
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
        # --- NY KOD BÖRJAR HÄR ---
        # 1. Bestäm antal kanaler baserat på modalitet
        # if mod == 'mri':
        #     channels = 1
        # elif mod == 'amy':
        #     channels = 4
        # else:
        #     channels = 3 # Fallback om du lägger till något annat (t.ex. RGB-video)

        print(f"  initializing backbone for {mod} (channels={channels})...", flush=True)
        
        # 2. Skapa Backbone och skicka med in_channels
        # OBS: Detta kräver att du har uppdaterat video_resnet18_features enligt min tidigare instruktion!
        backbone = base_architecture_to_features[args.net](
            pretrained = not args.disable_pretrained, 
            in_channels = channels
        )
        # --- NY KOD SLUTAR HÄR ---
        # # 1. Skapa Backbone
        # # Här antar vi samma arkitektur för alla, men du kan ha en if-sats om du vill ha olika
        # print(f"  initializing backbone for {mod}...", flush=True)
        # backbone = base_architecture_to_features[args.net](pretrained = not args.disable_pretrained)
        
        # 2. Hitta output channels
        backbone_out_channels = _get_backbone_channels(args, backbone) # (Använd hjälpfunktionen från förra svaret)

        # 3. Skapa Add-on layer
        # Här kan du välja om alla ska ha samma antal prototyper eller olika
        # T.ex. args.num_features delat på antal modaliteter?
        # För nu kör vi args.num_features per modalitet.
        add_on, n_protos = _create_add_on_layer(backbone_out_channels, args.num_features)
        
        # 4. Lägg in i ModuleDicts
        backbones[mod] = backbone
        add_ons[mod] = add_on
        
        total_prototypes += n_protos
        prototypes_per_modality[mod] = n_protos

    # 5. Pooling (delad)
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
        
    # Returnera ModuleDicts istället för enskilda lager
    return backbones, add_ons, pool_layer, classification_layer, total_prototypes






