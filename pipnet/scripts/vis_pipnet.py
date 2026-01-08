#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:18:57 2023

@author: lisadesanti
Updated for Multimodal PIPNet (MRI+AMY) - FIX FOR FILE NAMES
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
import torch.nn.functional as F
import torch.utils.data
import os
from PIL import Image, ImageDraw as D
import monai.transforms as transforms
import torchvision
from plot_utils import plot_3d_slices, plot_rgb_slices, generate_rgb_array, plot_atlas_overlay
import random
from tqdm.auto import tqdm
import pandas as pd


_image_cache = {}

def load_and_preprocess_image(img_path, args, modality="mri", use_cache=True):
    cache_key = f"{img_path}_{modality}"
    if use_cache and cache_key in _image_cache:
        return _image_cache[cache_key]

    if pd.isna(img_path) or str(img_path).lower() == 'nan':
        return torch.zeros((1, 3, args.slices, args.rows, args.cols))

    try:
        vol = np.load(img_path).astype(np.float32)
        
        img_min = vol.min()
        img_max = vol.max()
        if img_max > img_min:
            vol = (vol - img_min) / (img_max - img_min)
            
        if vol.ndim == 3: 
            vol = np.expand_dims(vol, axis=0)
            vol = np.repeat(vol, 3, axis=0)
            
        elif vol.ndim == 4:
            vol = np.transpose(vol, (3, 0, 1, 2))
            if vol.shape[0] >= 3:
                vol = vol[:3, ...]
            else:
                pad = np.zeros_like(vol[0:1])
                vol = np.concatenate([vol, pad, pad], axis=0)[:3]
                
        img_tensor = torch.from_numpy(vol)
        img_tensor = transforms.Resize(spatial_size=(args.slices, args.rows, args.cols))(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)

        if use_cache:
            _image_cache[cache_key] = img_tensor

        return img_tensor
        
    except Exception as e:
        print(f"[ERROR] Could not load {img_path}: {e}")
        return torch.zeros((1, 3, args.slices, args.rows, args.cols))


def create_edge_mask_spatial(img_shape, d_min, d_max, h_min, h_max, w_min, w_max):
    depth, height, width = img_shape[2], img_shape[3], img_shape[4]
    d_max = min(d_max, depth)
    h_max = min(h_max, height)
    w_max = min(w_max, width)
    
    mask = torch.zeros((1, 1, depth, height, width), dtype=torch.bool)
    erosion = torch.zeros((1, 1, depth, height, width), dtype=torch.bool)

    mask[:, :, d_min:d_max, h_min:h_max, w_min:w_max] = True
    erosion[:, :, d_min+1:d_max-1, h_min+1:h_max-1, w_min+1:w_max-1] = True
    
    edge_mask = mask & (~erosion)
    return edge_mask


def clear_image_cache():
    global _image_cache
    _image_cache.clear()

# --- UPDATE: Robustare regex för Subjekt ---
# Letar efter mönstret "tre siffror, _S_, fyra siffror" oavsett var det är i pathen
_pattern_subj = re.compile(r"(\d{3}_S_\d{4})")


def get_patch_size(args):
    patch_z = round(args.img_shape[0]/args.dshape)
    patch_y = round(args.img_shape[1]/args.hshape)
    patch_x = round(args.img_shape[2]/args.hshape)
    skip_z = round((args.img_shape[0] - patch_z) / (args.dshape-1))
    skip_y = round((args.img_shape[1] - patch_y) / (args.hshape-1))
    skip_x = round((args.img_shape[2] - patch_x) / (args.wshape-1))
    return (patch_z, patch_y, patch_x), skip_z, skip_y, skip_x


def get_img_coordinates(slices, rows, cols, softmaxes_shape, patchsize, skip_z, skip_y, skip_x, d_idx, h_idx, w_idx):
    d_min = d_idx*skip_z
    d_max = min(slices, d_idx*skip_z + patchsize[0])
    h_min = h_idx*skip_y
    h_max = min(rows, h_idx*skip_y + patchsize[1])
    w_min = w_idx*skip_x
    w_max = min(cols, w_idx*skip_x + patchsize[2])                                    
    
    if d_idx == softmaxes_shape[2]-1: d_max = slices
    if h_idx == softmaxes_shape[3]-1: h_max = rows
    if w_idx == softmaxes_shape[4]-1: w_max = cols
    if d_max == slices: d_min = slices-patchsize[0]
    if h_max == rows: h_min = rows-patchsize[1]
    if w_max == cols: w_min = cols-patchsize[2]

    return d_min, d_max, h_min, h_max, w_min, w_max


@torch.no_grad()                    
def visualize_topk(net, projectloader, num_classes, device, foldername, args, save: bool, k=10, plot=False):
    
    clear_image_cache()

    print(f"[INFO] Visualizing prototypes for topk in {os.path.join(args.log_dir, foldername)}...", flush = True)
    dir = os.path.join(args.log_dir, foldername)
    if save or plot:
        if not os.path.exists(dir): os.makedirs(dir)
    save_dir = os.path.join(dir, "saved")
    if save and not os.path.exists(save_dir): os.makedirs(save_dir)
    plot_dir = os.path.join(dir, "plots")
    if plot and not os.path.exists(plot_dir): os.makedirs(plot_dir)

    near_imgs_dirs = dict()
    saved = dict()
    tensors_per_prototype = dict()
    img_prototype = dict()
    proto_coord = dict()
    
    num_prototypes = net.module._classification.weight.shape[1]
    
    for p in range(num_prototypes):
        saved[p] = 0
        tensors_per_prototype[p] = []
        img_prototype[p] = []
        proto_coord[p] = []
    
    patchsize, skip_z, skip_y, skip_x = get_patch_size(args)
    dataset_paths = projectloader.dataset.X_paths
    modalities = list(dataset_paths.keys()) 

    modality_offsets = {}
    current_offset = 0
    if hasattr(net.module, 'modalities'):
        modalities = net.module.modalities
    
    for mod in modalities:
        add_on_module = net.module._add_ons[mod]
        num_protos_mod = 0
        
        # --- ROBUST PROTOTYPE COUNT ---
        for m in add_on_module.modules():
            if isinstance(m, torch.nn.Conv3d):
                num_protos_mod = m.out_channels
                break
        
        if num_protos_mod == 0: 
            print(f"[WARN] Could not detect prototypes for {mod}, using fallback 512.")
            num_protos_mod = 512
            
        modality_offsets[mod] = (current_offset, current_offset + num_protos_mod)
        current_offset += num_protos_mod
        
    def get_modality_for_proto(p_idx):
        for mod, (start, end) in modality_offsets.items():
            if start <= p_idx < end:
                return mod, p_idx - start
        return None, 0

    net.eval()
    classification_weights = net.module._classification.weight

    desc_text = f"Search top{k}"
    img_iter = tqdm(enumerate(projectloader), total=len(projectloader), desc=desc_text, mininterval=2., ncols=0)
    
    topks = dict()
    
    for i, (xs, ms, ys) in img_iter:
        ys = ys.to(device)
        xs = {key: val.to(device) for key, val in xs.items()}
        ms = {key: val.to(device) for key, val in ms.items()} if ms is not None else None

        with torch.no_grad():
            _, pooled, _ = net(xs, masks=ms, inference = True)
            pooled = pooled.squeeze(0)
            
            for p in range(pooled.shape[0]):
                # if True:
                c_weight = torch.max(classification_weights[:, p])
                # ignore prototypes that are not relevant to any class
                if c_weight > 1e-3: 
                    if p not in topks.keys(): topks[p] = []
                    
                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item())) 
                    else:
                        topks[p] = sorted(topks[p], key = lambda tup: tup[1], reverse = True)
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            if random.choice([0, 1]) > 0:
                                topks[p][-1] = (i, pooled[p].item())

    alli = [] 
    prototypes_not_used = []
    
    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if score > 0.0001: found = True
        
        if not found:
            prototypes_not_used.append(p)
            
    abstained = 0
    
    desc_text = f"Localize"
    img_iter = tqdm(enumerate(projectloader), total=len(projectloader), desc=desc_text, mininterval=2., ncols=0)
    
    for i, (xs, ms, ys) in img_iter:
        if i in alli:
            ys = ys.to(device)
            xs = {key: val.to(device) for key, val in xs.items()}
            ms = {key: val.to(device) for key, val in ms.items()} if ms is not None else None
            
            with torch.no_grad():
                softmaxes_dict, pooled, out = net(xs, masks=ms, inference = True)             
                outmax = torch.amax(out, dim=1)[0]
            
            for p in topks.keys():
                if p not in prototypes_not_used:
                    for idx, score in topks[p]:
                        if idx == i:
                            if outmax.item() == 0.: abstained += 1
                            
                            target_mod, local_p = get_modality_for_proto(p)
                            if target_mod is None: continue
                            
                            softmaxes = softmaxes_dict[target_mod]
                            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0) 
                            max_per_prototype_hw, max_idx_per_prototype_hw = torch.max(max_per_prototype, dim=1) 
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype_hw, dim=1) 
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) 
                            
                            d_idx = max_idx_per_prototype_hw[local_p, max_idx_per_prototype_h[local_p, max_idx_per_prototype_w[local_p]], max_idx_per_prototype_w[local_p]].item()
                            h_idx = max_idx_per_prototype_h[local_p, max_idx_per_prototype_w[local_p]].item()
                            w_idx = max_idx_per_prototype_w[local_p].item()
                            
                            img_path = dataset_paths[target_mod][i]
                            img_tensor = load_and_preprocess_image(img_path, args, modality=target_mod, use_cache=True) 
                            
                            ps_coord = get_img_coordinates(args.slices, args.rows, args.cols, softmaxes.shape, patchsize, skip_z, skip_y, skip_x, d_idx, h_idx, w_idx)
                            d_min, d_max, h_min, h_max, w_min, w_max = ps_coord
                            
                            img_tensor_patch = img_tensor[0, :, d_min:d_max, h_min:h_max, w_min:w_max]
                                    
                            saved[p]+=1
                            tensors_per_prototype[p].append(img_tensor_patch.numpy())
                            img_prototype[p].append((img_path, target_mod))
                            proto_coord[p].append(ps_coord)
                                

    print("Abstained: ", abstained, flush = True)
    all_tensors = []
    
    for p in tqdm(range(num_prototypes), desc="Processing prototypes"):
        if saved[p] > 0:
            text = "f_" + str(args.current_fold) + "_p_" + str(p)
            
            for (img_name, mod), tensor, ps_coord in zip(img_prototype[p], tensors_per_prototype[p], proto_coord[p]):
                img_tensor = load_and_preprocess_image(img_name, args, modality=mod, use_cache=False)
                d_min, d_max, h_min, h_max, w_min, w_max = ps_coord

                spatial_mask = create_edge_mask_spatial(img_tensor.shape, d_min, d_max, h_min, h_max, w_min, w_max)
                img_tensor[:, 0:1][spatial_mask] = 1.0
                img_tensor[:, 1:2][spatial_mask] = 1.0
                img_tensor[:, 2:3][spatial_mask] = 1.0
                
                image = img_tensor.detach().cpu().numpy() 

                # --- FIX: RÄTT NAMNHANTERING OAVSETT MRI/AMY ---
                try:
                    img_str = str(img_name)
                    # 1. Hitta Subjekt (XXX_S_XXXX)
                    match_subj = _pattern_subj.search(img_str)
                    if match_subj:
                        subj = match_subj.group(1) # Hela matchningen XXX_S_XXXX
                    else:
                        subj = "UnkSubj"
                    
                    # 2. Hitta Exam ID (Filnamn minus extension)
                    # Detta fungerar för BÅDE "d26eb..." och "I1598943"
                    base_name = os.path.basename(img_str)
                    exam, _ = os.path.splitext(base_name)
                    
                except:
                    subj, exam = "unknown", "unknown"
                # -----------------------------------------------
                    
                ps_name = text + "_" + mod + "_" + subj + "_" + exam
                ps_patch_name = text + "_" + mod + "_patch_" + subj + "_" + exam
                
                plot_name = plot_dir + "/" + ps_name + ".png"
                plot_patch_name = plot_dir + "/" + ps_patch_name + ".png"
                
                if plot:
                    try:
                        plot_rgb_slices(image[0,:,:,:,:], title = f"Proto {p} ({mod})", num_columns = 10, bottom=True, save_path=plot_name)   
                        plot_rgb_slices(tensor[:,:,:,:], title = f"Proto {p} Patch", num_columns = 6, bottom=True, save_path=plot_patch_name)
                    except Exception as e:
                        print(f"Error plotting {ps_name}: {e}")
                    
                if save:
                    np.save(os.path.join(save_dir, ps_name), image[0,:,:,:,:])
                    np.save(os.path.join(save_dir, ps_patch_name), tensor[:,:,:,:])
                        
                if saved[p] >= k:
                    all_tensors += tensors_per_prototype[p]

    return topks, img_prototype, proto_coord


def plot_local_explanation(xs, local_explanation, modality_offsets, title="", save_path=None):
    if not isinstance(xs, dict):
        print("[WARN] plot_local_explanation received non-dict xs.")
        return

    modalities_to_plot = {} 
    num_ps = len(local_explanation.keys())
    rgb_colors = generate_rgb_array(num_ps)
    
    for i, (ps_idx, data) in enumerate(local_explanation.items()):
        
        target_mod = None
        for mod, (start, end) in modality_offsets.items():
            if start <= ps_idx < end:
                target_mod = mod
                break
        
        if target_mod is None:
            print(f"[WARN] Prototype index {ps_idx} unknown offset.")
            continue

        if target_mod not in modalities_to_plot:
            modalities_to_plot[target_mod] = []
        modalities_to_plot[target_mod].append((ps_idx, data, rgb_colors[i]))

    for mod, protos in modalities_to_plot.items():
        if mod not in xs: continue
        
        img_tensor = xs[mod].clone().detach().cpu()
        
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1, 1)
        elif img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3, :, :, :]
            
        ps_scores = []
        for (ps_idx, (ps_coord, ps_score), color) in protos:
            ps_scores.append((f"P{ps_idx}: {ps_score:.2f}", color))
            d_min, d_max, h_min, h_max, w_min, w_max = ps_coord
            
            mask = create_edge_mask_spatial(img_tensor.shape, d_min, d_max, h_min, h_max, w_min, w_max)
            img_tensor[:, 0:1][mask] = float(color[0])
            img_tensor[:, 1:2][mask] = float(color[1])
            img_tensor[:, 2:3][mask] = float(color[2])

        if save_path:
            base, ext = os.path.splitext(save_path)
            mod_save_path = f"{base}_{mod}{ext}"
            try:
                plot_rgb_slices(img_tensor[0].numpy(), title=f"{title} \n({mod})", legend=ps_scores, save_path=mod_save_path)
            except Exception as e:
                print(f"Error plotting local expl {mod_save_path}: {e}")