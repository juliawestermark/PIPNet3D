#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:18:57 2023

@author: lisadesanti
Updated for Multimodal PIPNet (MRI+AMY) - FIX FOR DATASET CROP
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

# Regex för att hitta Subject ID i filnamn
_pattern_subj = re.compile(r"(\d{3}_S_\d+)")

def create_edge_mask_spatial(img_shape, d_min, d_max, h_min, h_max, w_min, w_max):
    # img_shape är (1, 1, D, H, W) eller (1, 3, D, H, W)
    depth, height, width = img_shape[-3], img_shape[-2], img_shape[-1]
    
    # Säkra upp så vi inte går utanför bilden
    d_max = min(d_max, depth)
    h_max = min(h_max, height)
    w_max = min(w_max, width)
    
    mask = torch.zeros((1, 1, depth, height, width), dtype=torch.bool)
    erosion = torch.zeros((1, 1, depth, height, width), dtype=torch.bool)

    # Kolla att vi har valida intervall
    if d_min >= d_max or h_min >= h_max or w_min >= w_max:
        return mask # Returnera tom mask om koordinaterna är fel

    mask[:, :, d_min:d_max, h_min:h_max, w_min:w_max] = True
    
    # Erosion för att skapa bara en kant (outline)
    # Vi måste vara försiktiga så vi inte eroderar bort allt om boxen är liten
    if (d_max - d_min) > 2 and (h_max - h_min) > 2 and (w_max - w_min) > 2:
        erosion[:, :, d_min+1:d_max-1, h_min+1:h_max-1, w_min+1:w_max-1] = True
    
    edge_mask = mask & (~erosion)
    return edge_mask

# --- NY: Dynamisk patch-storlek beroende på bildens faktiska storlek ---
def get_patch_size_dynamic(current_img_shape, args):
    """
    Räknar ut patch-storlek baserat på den AKTUELLA bildens dimensioner,
    inte de globala argumenten.
    
    current_img_shape: (D, H, W) tuple
    """
    d, h, w = current_img_shape
    
    # args.dshape etc är nätverkets output-grid (t.ex. 1x1x1 eller 7x7x7)
    # Vi delar bildens storlek med grid-storleken för att se hur många pixlar en "prototyp" täcker.
    patch_z = round(d / args.dshape)
    patch_y = round(h / args.hshape)
    patch_x = round(w / args.hshape) # OBS: args.hshape används ofta för width också i PIPNet-kod, kolla om du har args.wshape
    
    # Skydd mot division by zero om grid är 1
    denom_z = max(1, args.dshape - 1)
    denom_y = max(1, args.hshape - 1)
    denom_x = max(1, args.wshape - 1) # eller hshape om wshape saknas

    skip_z = round((d - patch_z) / denom_z)
    skip_y = round((h - patch_y) / denom_y)
    skip_x = round((w - patch_x) / denom_x)
    
    return (patch_z, patch_y, patch_x), skip_z, skip_y, skip_x

def get_img_coordinates(curr_slices, curr_rows, curr_cols, softmaxes_shape, patchsize, skip_z, skip_y, skip_x, d_idx, h_idx, w_idx):
    """
    Beräknar koordinater baserat på bildens FAKTISKA storlek (curr_...)
    """
    d_min = d_idx * skip_z
    d_max = min(curr_slices, d_idx * skip_z + patchsize[0])
    
    h_min = h_idx * skip_y
    h_max = min(curr_rows, h_idx * skip_y + patchsize[1])
    
    w_min = w_idx * skip_x
    w_max = min(curr_cols, w_idx * skip_x + patchsize[2])                                    
    
    # Justera för sista indexet (för att täcka kanten)
    if d_idx == softmaxes_shape[2]-1: d_max = curr_slices
    if h_idx == softmaxes_shape[3]-1: h_max = curr_rows
    if w_idx == softmaxes_shape[4]-1: w_max = curr_cols
    
    # Om patchen hamnar utanför, dra in den
    if d_max == curr_slices: d_min = max(0, curr_slices - patchsize[0])
    if h_max == curr_rows: h_min = max(0, curr_rows - patchsize[1])
    if w_max == curr_cols: w_min = max(0, curr_cols - patchsize[2])

    return d_min, d_max, h_min, h_max, w_min, w_max

@torch.no_grad()                    
def visualize_topk(net, projectloader, num_classes, device, foldername, args, save: bool, k=10, plot=False, threshold=None):
    
    print(f"[INFO] Visualizing prototypes for topk in {os.path.join(args.log_dir, foldername)}...", flush = True)
    dir = os.path.join(args.log_dir, foldername)
    if save or plot:
        if not os.path.exists(dir): os.makedirs(dir)
    save_dir = os.path.join(dir, "saved")
    if save and not os.path.exists(save_dir): os.makedirs(save_dir)
    plot_dir = os.path.join(dir, "plots")
    if plot and not os.path.exists(plot_dir): os.makedirs(plot_dir)

    saved = dict()
    tensors_per_prototype = dict()
    img_prototype_info = dict() 
    proto_coord = dict()
    
    num_prototypes = net.module._classification.weight.shape[1]
    
    for p in range(num_prototypes):
        saved[p] = 0
        tensors_per_prototype[p] = []
        img_prototype_info[p] = []
        proto_coord[p] = []
    
    # dataset_paths = projectloader.dataset.X_paths

    # ---------------------------------------------------------
    # FIX FÖR SUBSET (När vi kör snabb-test)
    # ---------------------------------------------------------
    
    # Kolla om datasetet är en Subset (dvs. krympt version)
    if isinstance(projectloader.dataset, torch.utils.data.Subset):
        # 1. Hämta original-datasetet som gömmer sig inuti
        original_dataset = projectloader.dataset.dataset
        
        # 2. Hämta alla paths från originalet
        full_paths = original_dataset.X_paths
        
        # 3. Filtrera ut BARA de paths som ingår i vår subset
        # (Subset.indices talar om vilka bilder vi valt ut)
        dataset_paths = [full_paths[i] for i in projectloader.dataset.indices]
        
    else:
        # Vanligt fall (Hela datasetet)
        dataset_paths = projectloader.dataset.X_paths
    # ---------------------------------------------------------
    
    # --- SETUP MODALITY OFFSETS ---
    modalities = list(dataset_paths.keys())
    modality_offsets = {}
    current_offset = 0
    if hasattr(net.module, 'modalities'):
        modalities = net.module.modalities
    
    for mod in modalities:
        add_on_module = net.module._add_ons[mod]
        num_protos_mod = 0
        for m in add_on_module.modules():
            if isinstance(m, torch.nn.Conv3d):
                num_protos_mod = m.out_channels
                break
        if num_protos_mod == 0: num_protos_mod = 512
        modality_offsets[mod] = (current_offset, current_offset + num_protos_mod)
        current_offset += num_protos_mod
        
    def get_modality_for_proto(p_idx):
        for mod, (start, end) in modality_offsets.items():
            if start <= p_idx < end:
                return mod, p_idx - start
        return None, 0
    # ------------------------------

    net.eval()
    classification_weights = net.module._classification.weight

    # === STEG 1: SÖK EFTER TOP K ===
    desc_text = f"Search top{k}"
    img_iter = tqdm(enumerate(projectloader), total=len(projectloader), desc=desc_text, mininterval=2., ncols=0)
    topks = dict()
    
    for i, (xs, ms, ys) in img_iter:
        ys = ys.to(device)
        xs = {key: val.to(device) for key, val in xs.items()}
        ms = {key: val.to(device) for key, val in ms.items()} if ms is not None else None

        with torch.no_grad():
            _, pooled, _ = net(xs, masks=ms, inference = True, threshold=threshold)
            pooled = pooled.squeeze(0)
            
            for p in range(pooled.shape[0]):
                c_weight = torch.max(classification_weights[:, p])
                if c_weight > 1e-3: 
                    if p not in topks.keys(): topks[p] = []
                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item())) 
                    else:
                        topks[p] = sorted(topks[p], key = lambda tup: tup[1], reverse = True)
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            if random.choice([0, 1]) > 0: topks[p][-1] = (i, pooled[p].item())

    alli = [] 
    prototypes_not_used = []
    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if score > 0.0001: found = True
        if not found: prototypes_not_used.append(p)
            
    abstained = 0
    
    # === STEG 2: LOKALISERA OCH SPARA PATCHES ===
    desc_text = f"Localize"
    img_iter = tqdm(enumerate(projectloader), total=len(projectloader), desc=desc_text, mininterval=2., ncols=0)
    
    for i, (xs, ms, ys) in img_iter:
        if i in alli:
            ys = ys.to(device)
            xs = {key: val.to(device) for key, val in xs.items()}
            ms = {key: val.to(device) for key, val in ms.items()} if ms is not None else None
            
            with torch.no_grad():
                softmaxes_dict, pooled, out = net(xs, masks=ms, inference = True, threshold=threshold)             
                outmax = torch.amax(out, dim=1)[0]
            
            for p in topks.keys():
                if p not in prototypes_not_used:
                    for idx, score in topks[p]:
                        if idx == i:
                            if outmax.item() == 0.: abstained += 1
                            
                            target_mod, local_p = get_modality_for_proto(p)
                            if target_mod is None: continue
                            
                            img_tensor = xs[target_mod].cpu()

                            # 1. Hoppa över om bilden är tom (missing modality)
                            if img_tensor.max() <= img_tensor.min() + 1e-9:
                                continue

                            softmaxes = softmaxes_dict[target_mod]
                            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0) 
                            max_per_prototype_hw, max_idx_per_prototype_hw = torch.max(max_per_prototype, dim=1) 
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype_hw, dim=1) 
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) 
                            
                            d_idx = max_idx_per_prototype_hw[local_p, max_idx_per_prototype_h[local_p, max_idx_per_prototype_w[local_p]], max_idx_per_prototype_w[local_p]].item()
                            h_idx = max_idx_per_prototype_h[local_p, max_idx_per_prototype_w[local_p]].item()
                            w_idx = max_idx_per_prototype_w[local_p].item()
                            
                            if img_tensor.shape[1] == 1:
                                img_tensor = img_tensor.repeat(1, 3, 1, 1, 1)
                            
                            # --- 2. HÄMTA FAKTISKA DIMENSIONER ---
                            curr_slices = img_tensor.shape[2]
                            curr_rows = img_tensor.shape[3]
                            curr_cols = img_tensor.shape[4]
                            
                            # --- 3. BERÄKNA PATCH SIZE FÖR DENNA BILD ---
                            # Använd den nya funktionen här!
                            patchsize, skip_z, skip_y, skip_x = get_patch_size_dynamic((curr_slices, curr_rows, curr_cols), args)
                            
                            # --- 4. BERÄKNA KOORDINATER MED LOKALA DIMENSIONER ---
                            # Skicka in curr_slices etc istället för args.slices
                            ps_coord = get_img_coordinates(
                                curr_slices, curr_rows, curr_cols, 
                                softmaxes.shape, 
                                patchsize, skip_z, skip_y, skip_x, 
                                d_idx, h_idx, w_idx
                            )
                            
                            d_min, d_max, h_min, h_max, w_min, w_max = ps_coord
                            
                            # --- 5. VALIDERA ATT PATCHEN ÄR GILTIG ---
                            if (d_max <= d_min) or (h_max <= h_min) or (w_max <= w_min):
                                continue

                            img_path = dataset_paths[target_mod][i]
                            img_tensor_patch = img_tensor[0, :, d_min:d_max, h_min:h_max, w_min:w_max]
                                    
                            saved[p]+=1
                            tensors_per_prototype[p].append(img_tensor_patch.numpy())
                            img_prototype_info[p].append((i, target_mod, img_path))
                            proto_coord[p].append(ps_coord)
                                

    print("Abstained: ", abstained, flush = True)
    
    # === STEG 3: PLOTTA ===
    for p in tqdm(range(num_prototypes), desc="Processing prototypes"):
        if saved[p] > 0:
            text = "f_" + str(args.current_fold) + "_p_" + str(p)
            
            for (dataset_idx, mod, img_name), tensor, ps_coord in zip(img_prototype_info[p], tensors_per_prototype[p], proto_coord[p]):
                
                try:
                    inputs_dict, _, _ = projectloader.dataset[dataset_idx]
                    img_tensor_raw = inputs_dict[mod]
                    
                    if img_tensor_raw.max() <= img_tensor_raw.min() + 1e-9:
                        continue

                    img_tensor = img_tensor_raw.unsqueeze(0) 
                    if img_tensor.shape[1] == 1:
                        img_tensor = img_tensor.repeat(1, 3, 1, 1, 1)
                        
                except Exception as e:
                    print(f"[ERROR] Could not fetch index {dataset_idx}: {e}")
                    continue

                d_min, d_max, h_min, h_max, w_min, w_max = ps_coord
                
                # Använd mask-funktionen. Den borde nu vara säker eftersom koordinaterna
                # är baserade på bildens verkliga storlek.
                spatial_mask = create_edge_mask_spatial(img_tensor.shape, d_min, d_max, h_min, h_max, w_min, w_max)

                img_tensor[:, 0:1][spatial_mask] = 1.0  # Röd kanal = Max
                img_tensor[:, 1:2][spatial_mask] = 0.0  # Grön kanal = 0
                img_tensor[:, 2:3][spatial_mask] = 0.0  # Blå kanal = 0
                
                image = img_tensor.detach().cpu().numpy() 

                try:
                    img_str = str(img_name)
                    match_subj = _pattern_subj.search(img_str)
                    subj = match_subj.group(1) if match_subj else "UnkSubj"
                    base_name = os.path.basename(img_str)
                    exam, _ = os.path.splitext(base_name)
                except:
                    subj, exam = "unknown", "unknown"
                    
                ps_name = text + "_" + mod + "_" + subj + "_" + exam
                ps_patch_name = text + "_" + mod + "_patch_" + subj + "_" + exam
                
                plot_name = plot_dir + "/" + ps_name + ".png"
                plot_patch_name = plot_dir + "/" + ps_patch_name + ".png"
                
                if plot:
                    try:
                        if np.all(image == 0) or np.any(np.array(image.shape) == 0):
                             continue

                        plot_rgb_slices(image[0,:,:,:,:], title = f"Proto {p} ({mod})", num_columns = 10, bottom=True, save_path=plot_name)   
                        plot_rgb_slices(tensor[:,:,:,:], title = f"Proto {p} Patch", num_columns = 6, bottom=True, save_path=plot_patch_name)
                    except Exception as e:
                        print(f"Error plotting {ps_name}: {e}")
                    
                if save:
                    np.save(os.path.join(save_dir, ps_name), image[0,:,:,:,:])
                    np.save(os.path.join(save_dir, ps_patch_name), tensor[:,:,:,:])

    return topks, img_prototype_info, proto_coord

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
        
        if target_mod is None: continue

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