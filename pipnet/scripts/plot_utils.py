#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:16:01 2024

@author: lisadesanti
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import torch
import argparse
import pickle
import random
import torch.optim
from datetime import datetime
import matplotlib.cm as cm
import os



def generate_rgb_array(n):
    
    # Generate a list of n distinct colors using matplotlib's 'tab10' colormap
    cmap = plt.get_cmap('tab10')
    color_list = [cmap(i)[:3] for i in np.linspace(0, 1, n)]

    # Convert the list to a NumPy array
    rgb_array = np.array(color_list)

    return rgb_array 

  
def plot_3d_slices(data, num_columns=10, cmap="gray", title=False, data_min=False, data_max=False, save_path=False, bottom=False):
    
    depth = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]
    
    if not(data_min) or not(data_max):
        data_min = data.min()
        data_max = data.max()
    
    r, num_rows = math.modf(depth/num_columns)
    num_rows = int(num_rows)
    if num_rows == 0:
        num_columns = int(r*num_columns)
        num_rows +=1
        r = 0
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows))
        add = np.zeros((new_im, width, height), dtype=type(data[0,0,0]))
        data = np.concatenate((data, add), axis=0)
        num_rows +=1
    
    data = np.reshape(data, (num_rows, num_columns, width, height))

    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    
    f, axarr = plt.subplots(rows_data, columns_data, figsize=(fig_width, fig_height), gridspec_kw={"height_ratios": heights},)
        
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img = axarr[i, j].imshow(data[i][j], cmap=cmap, vmin=data_min, vmax=data_max)
                axarr[i, j].axis("off")
            else:
                img = axarr[j].imshow(data[i][j], cmap=cmap, vmin=data_min, vmax=data_max)
                axarr[j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9)
    
    if title:
        f.suptitle(title)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    plt.close(f)
    

def plot_rgb_slices(data, num_columns=10, title=False, save_path=False, bottom=False, legend=False):
    """
    Plot all the slices of a 3D volume (both gray-scale or RGB) stored in a 
    numpy array.
    Takes:
        - data: np.array, expected dimension (channels, slices, rows, columns)
        - num_columns
        - title
        - data_min
        - data_max
        - save_path
    """
    
    channels, depth, width, height = data.shape
    
    r, num_rows = math.modf(depth/num_columns)
    num_rows = int(num_rows)
    if num_rows == 0:
        num_columns = int(r*num_columns)
        num_rows +=1
        r = 0
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows))
        add = np.zeros((channels, new_im, width, height), dtype=float)
        data = np.concatenate((data, add), axis=1)
        num_rows +=1
    
    data = np.reshape(data, (channels, num_rows, num_columns, width, height))
    data = np.transpose(data, (1, 2, 3, 4, 0))

    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    
    f, axarr = plt.subplots(rows_data, columns_data, figsize=(fig_width, fig_height), gridspec_kw={"height_ratios": heights}, )
        
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img = axarr[i, j].imshow(data[i][j], vmin=0., vmax=1.)
                axarr[i, j].axis("off")
            else:
                img = axarr[j].imshow(data[i][j], vmin=0., vmax=1.)
                axarr[j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9)
    
    if title:
        if bottom:
            f.suptitle(title, fontsize="large", y=0., va="top", color="gray")
        else:
            f.suptitle(title)
            
    if save_path:
        f.savefig(save_path, bbox_inches='tight')
        
    plt.show()
    plt.close(f)


def plot_atlas_overlay(data, data_atlas, num_columns=10, title=False,):
    
    depth = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]
    
    r, num_rows = math.modf(depth/num_columns)
    num_rows = int(num_rows)
    if num_rows == 0:
        num_columns = int(r*num_columns)
        num_rows +=1
        r = 0
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows))
        add = np.zeros((new_im, width, height), dtype=type(data[0,0,0]))
        data = np.concatenate((data, add), axis=0)
        data_atlas = np.concatenate((data_atlas, add), axis=0)
        num_rows +=1
    
    data = np.reshape(data, (num_rows, num_columns, width, height))
    data_atlas = np.reshape(data_atlas, (num_rows, num_columns, width, height))
    
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    
    f, axarr = plt.subplots(rows_data, columns_data, figsize=(fig_width, fig_height), gridspec_kw={"height_ratios": heights},);
    
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img1 = axarr[i, j].imshow(data[i][j], cmap=plt.cm.gray, alpha=0.7,)
                img2 = axarr[i, j].imshow(data_atlas[i][j], cmap=plt.cm.jet, alpha=0.3,)
                axarr[i, j].axis("off")
            else:
                img1 = axarr[j].imshow(data[i][j], cmap=plt.cm.gray, alpha=0.7,)
                img2 = axarr[j].imshow(data_atlas[i][j], cmap=plt.cm.jet, alpha=0.3,)
                axarr[j].axis("off")
    
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9)
    
    if title:
        f.suptitle(title)
    
    plt.show()
    plt.close(f)


def plot_proto_distribution_dynamic(proto_df, weights, modality_indices, log_dir, reference_volume=None):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Standardgränser
    max_d, max_h, max_w = 64, 64, 64
    
    # --- STEG 1: Rita 3 st tvärsnitt (Slices) för kontext ---
    if reference_volume is not None:
        print("[INFO] Plotting anatomical slices...", flush=True)
        vol = reference_volume
        d, h, w = vol.shape
        max_d, max_h, max_w = d, h, w
        
        # Hitta mitten
        mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
        
        # Skapa koordinat-nät för snitten
        dd, hh = np.meshgrid(np.arange(d), np.arange(h))
        dd_w, ww = np.meshgrid(np.arange(d), np.arange(w))
        hh_z, ww_z = np.meshgrid(np.arange(h), np.arange(w))

        # Vi använder contourf för att rita "bilder" i 3D-rymden.
        # cmap='gray' ger oss den klassiska MRI-looken.
        # alpha=0.4 gör dem genomskinliga så vi ser prototyper bakom.
        
        # Snitt 1: Sagittal (Sedd från sidan, fixerad Width)
        # zdir='z', offset=mid_w betyder att vi ritar planet vid z=mitten
        ax.contourf(dd, hh, vol[:, :, mid_w], zdir='z', offset=mid_w, cmap='gray', alpha=0.3)
        
        # Snitt 2: Coronal (Sedd framifrån, fixerad Depth)
        # zdir='x', offset=mid_d betyder att vi ritar planet vid x=mitten
        # Notera: Vi måste transponera matrisen ibland för att orienteringen ska bli rätt
        ax.contourf(vol[mid_d, :, :].T, hh_z, ww_z, zdir='x', offset=mid_d, cmap='gray', alpha=0.3)
        
        # Snitt 3: Axial (Sedd uppifrån, fixerad Height)
        # zdir='y', offset=mid_h
        ax.contourf(dd_w, vol[:, mid_h, :], ww, zdir='y', offset=mid_h, cmap='gray', alpha=0.3)
        
        # --- LÄGG TILL TEXT FÖR ORIENTERING ---
        # Detta är gissade standard-riktningar. Justera om din data är roterad annorlunda.
        # ax.text(0, mid_h, mid_w, "Posterior/Back", color='black', fontsize=10, weight='bold')
        # ax.text(max_d, mid_h, mid_w, "Anterior/Front", color='black', fontsize=10, weight='bold')
        # ax.text(mid_d, max_h, mid_w, "Superior/Top", color='black', fontsize=10)
        # ax.text(mid_d, mid_h, 0, "Right", color='black', fontsize=10)
        # ax.text(mid_d, mid_h, max_w, "Left", color='black', fontsize=10)

    # --- STEG 2: Rita Prototyperna ---
    print("[INFO] Plotting prototypes...", flush=True)
    mods = list(modality_indices.keys())
    colors = cm.rainbow(np.linspace(0, 1, len(mods)))
    mod_colors = {mod: color for mod, color in zip(mods, colors)}
    
    for idx, row in proto_df.iterrows():
        if row['detection_rate'] == 0: continue

        my_mod = None
        for mod, (start, end) in modality_indices.items():
            if start <= idx < end:
                my_mod = mod
                break
        if my_mod is None: continue

        w = weights[:, idx].max().item()
        
        if w > 1e-3: 
            c = mod_colors[my_mod]
            # Rita bollen. edgecolors='black' ger en tydlig kant så den "poppar" ut
            ax.scatter(row['mean_pcc_d'], row['mean_pcc_h'], row['mean_pcc_w'], 
                       color=c, s=w*600, alpha=1.0, edgecolors='black', linewidth=1.0)
            
            # Valfritt: Dra ett streck från bollen ner till "golvet" för att se djupet
            # ax.plot([row['mean_pcc_d'], row['mean_pcc_d']], 
            #         [row['mean_pcc_h'], row['mean_pcc_h']], 
            #         [0, row['mean_pcc_w']], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Depth (x)')
    ax.set_ylabel('Height (y)')
    ax.set_zlabel('Width (z)')
    
    ax.set_xlim(0, max_d)
    ax.set_ylim(0, max_h)
    ax.set_zlim(0, max_w)
    
    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=mod_colors[mod], marker='o', linestyle='') for mod in mods]
    ax.legend(custom_lines, [m.upper() for m in mods], loc='upper left')

    # --- STEG 3: SPARA FLERA VINKLAR ---
    views = [
        ('iso', 25, -45),      # Standard översikt
        ('top', 90, -90),      # Uppifrån
        ('side', 0, 0),        # Sidan
        ('front', 0, -90)      # Framifrån
    ]
    
    base_path = os.path.join(log_dir, "prototype_spatial_dist_slices")
    
    for name, elev, azim in views:
        ax.view_init(elev=elev, azim=azim)
        save_path = f"{base_path}_{name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved view '{name}' to {save_path}")

    plt.close(fig)