#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:24:11 2023

@author: lisadesanti
Updated for Modality-Specific Resizing
"""

import argparse
import os
import math
import numpy as np
import random
import pandas as pd

import torch
from torch import Tensor
import torch.optim
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms

from monai.transforms import (
    Compose,
    Resize,
    RandRotate,
    Affine,
    RandAffine,
    RandGaussianNoise,
    RandZoom,
    RepeatChannel,
)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from make_mm_dataset import load_npy_dataset


def crop_fixed_rectangular(image_volume, target_shape, threshold_percent=None):
    """
    Supersnabb Center Crop.
    Eftersom datan är 'Aligned' vet vi att hjärnan är i mitten.
    Vi behöver inte söka efter den.
    """
    
    # 1. Hämta dimensioner
    if image_volume.ndim == 4:
        # (C, D, H, W)
        vol_dims = image_volume.shape[1:]
    else:
        # (D, H, W)
        vol_dims = image_volume.shape

    d_in, h_in, w_in = vol_dims
    tgt_d, tgt_h, tgt_w = target_shape

    # 2. Beräkna mitten direkt (Matematiskt center)
    center_z = d_in // 2
    center_y = h_in // 2
    center_x = w_in // 2

    # 3. Beräkna start- och slutpunkter för crop-boxen
    z_start = center_z - tgt_d // 2
    y_start = center_y - tgt_h // 2
    x_start = center_x - tgt_w // 2
    
    z_end = z_start + tgt_d
    y_end = y_start + tgt_h
    x_end = x_start + tgt_w

    # 4. Hantera gränser (Om boxen går utanför bilden, eller bilden är mindre än boxen)
    # Source (Bildens koordinater)
    src_z_start = max(0, z_start); src_z_end = min(d_in, z_end)
    src_y_start = max(0, y_start); src_y_end = min(h_in, y_end)
    src_x_start = max(0, x_start); src_x_end = min(w_in, x_end)
    
    # Destination (Den nya boxens koordinater)
    dst_z_start = max(0, -z_start)
    dst_y_start = max(0, -y_start)
    dst_x_start = max(0, -x_start)
    
    # 5. Skapa och fyll volymen (Kopiera data)
    dtype = image_volume.dtype
    
    if image_volume.ndim == 4:
        c = image_volume.shape[0]
        out_vol = np.zeros((c, tgt_d, tgt_h, tgt_w), dtype=dtype)
        out_vol[:, 
                dst_z_start : dst_z_start + (src_z_end - src_z_start),
                dst_y_start : dst_y_start + (src_y_end - src_y_start),
                dst_x_start : dst_x_start + (src_x_end - src_x_start)] = \
        image_volume[:, src_z_start : src_z_end, src_y_start : src_y_end, src_x_start : src_x_end]
    else:
        out_vol = np.zeros((tgt_d, tgt_h, tgt_w), dtype=dtype)
        out_vol[dst_z_start : dst_z_start + (src_z_end - src_z_start),
                dst_y_start : dst_y_start + (src_y_end - src_y_start),
                dst_x_start : dst_x_start + (src_x_end - src_x_start)] = \
        image_volume[src_z_start : src_z_end, src_y_start : src_y_end, src_x_start : src_x_end]
        
    return out_vol


def print_split_statistics(df, split_name, modalities):
    total = len(df)
    print(f"\n{'='*10} Statistics for {split_name.upper()} Set {'='*10}")
    print(f"Total entries (rows): {total}")
    found_cols = []
    for mod in modalities:
        col_name = f"file_path_{mod}"
        if col_name in df.columns:
            count = df[col_name].apply(lambda x: pd.notna(x) and str(x).strip() != "").sum()
            percent = (count / total * 100) if total > 0 else 0
            print(f"  - Has {mod.upper().ljust(5)}: {count} ({percent:.1f}%)")
            found_cols.append(col_name)
        else:
            print(f"  - Has {mod.upper().ljust(5)}: 0 (Column '{col_name}' missing)")

    if found_cols:
        has_all_mask = pd.Series([True] * total, index=df.index)
        for col in found_cols:
            is_valid = df[col].apply(lambda x: pd.notna(x) and str(x).strip() != "")
            has_all_mask = has_all_mask & is_valid
        n_complete = has_all_mask.sum()
        pct_complete = (n_complete / total * 100) if total > 0 else 0
        print(f"  ------------------------")
        print(f"  - Complete Intersection ({'+'.join(modalities)}): {n_complete} ({pct_complete:.1f}%)")
    print("="*40)


def build_preprocessed_paths(df, dic_classes, modalities):
    paths_dict = {mod: [] for mod in modalities}
    labels = []
    for _, row in df.iterrows():
        labels.append(dic_classes[row["clinical_stage"]])
        for mod in modalities:
            col_name = f"file_path_{mod}" 
            path = row.get(col_name, pd.NA)
            paths_dict[mod].append(path)
    X_paths = {mod: np.array(paths) for mod, paths in paths_dict.items()}
    return X_paths, np.array(labels)


def shuffle_arrays(X_dict, y, seed=42):
    n_samples = len(y)
    idx = np.arange(n_samples)
    np.random.default_rng(seed).shuffle(idx)
    X_shuffled = {}
    for mod, paths in X_dict.items():
        X_shuffled[mod] = paths[idx]
    return X_shuffled, y[idx]


def get_mm_paths(
        directory_dataframe,
        dic_classes={"CN":0,"MCI":1,"AD":2},
        set_type='train',
        shuffle=True,
        n_fold=5,
        current_fold=1,
        test_split=0.2,
        seed=42,
        modalities=["mri"]
    ):
    """
    Multimodal split på SUBJEKT-NIVÅ.
    Garanterar att inget subjekt läcker mellan train/val/test.
    """

    # 1. Identifiera unika subjekt
    unique_subjects_df = directory_dataframe.drop_duplicates(subset=["individual_id"])
    unique_subjects = unique_subjects_df["individual_id"].values
    unique_labels = unique_subjects_df["clinical_stage"].values

    train_val_subj, test_subj, train_val_labels, _ = train_test_split(
        unique_subjects, unique_labels, test_size=test_split, stratify=unique_labels, random_state=seed, shuffle=True
    )

    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    split_generator = skf.split(train_val_subj, train_val_labels)
    
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(split_generator):
        if i == (current_fold - 1):
            train_idx, val_idx = t_idx, v_idx
            break
            
    if train_idx is None: raise ValueError(f"Invalid fold {current_fold}")

    final_train_subj = train_val_subj[train_idx]
    final_val_subj   = train_val_subj[val_idx]

    X_train_df = directory_dataframe[directory_dataframe["individual_id"].isin(final_train_subj)].reset_index(drop=True)
    X_val_df   = directory_dataframe[directory_dataframe["individual_id"].isin(final_val_subj)].reset_index(drop=True)
    X_test_df  = directory_dataframe[directory_dataframe["individual_id"].isin(test_subj)].reset_index(drop=True)

    X_train, y_train = build_preprocessed_paths(X_train_df, dic_classes, modalities)
    X_val, y_val     = build_preprocessed_paths(X_val_df, dic_classes, modalities)
    X_test, y_test   = build_preprocessed_paths(X_test_df, dic_classes, modalities)

    if shuffle:
        X_train, y_train = shuffle_arrays(X_train, y_train, seed)
        X_val, y_val     = shuffle_arrays(X_val,   y_val,   seed)
        X_test, y_test   = shuffle_arrays(X_test,  y_test,  seed)

    info = {"n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test)}
    print(f"Fold {current_fold} ({set_type}): {len(y_train)} train, {len(y_val)} val, {len(y_test)} test images.")

    if set_type == "train": print_split_statistics(X_train_df, set_type, modalities); return X_train, y_train, info
    elif set_type == "val": print_split_statistics(X_val_df, set_type, modalities); return X_val, y_val, info
    elif set_type == "test": print_split_statistics(X_test_df, set_type, modalities); return X_test, y_test, info
    
    return X_train, y_train, info


class AugSupervisedDataset(torch.utils.data.Dataset):
    # ÄNDRING: target_shapes istället för img_shape
    def __init__(self, X_paths, y, dic_classes, target_shapes, mod_shape, transform=None):
        self.X_paths = X_paths
        self.y = y
        self.transform = transform # Detta är nu en DICTIONARY: {modality: transform}
        self.img_labels = y
        self.classes = list(dic_classes.keys())
        self.class_to_idx = dic_classes
        self.modalities = list(X_paths.keys())
        self.target_shapes = target_shapes # Dict: {mod: (D, H, W)}
        self.mod_shape = mod_shape # Dict: {mod: (D, H, W)} (Originalstorlek innan resize)

    def __len__(self):
        return len(self.y)

    def _get_empty_volume(self, modality):
        # Hämtar target shape för denna modalitet för att skapa rätt tom tensor
        t_shape = self.target_shapes[modality]
        return torch.zeros((3, *t_shape), dtype=torch.float32)

    def _load_volume(self, path, modality):
        if pd.isna(path) or str(path).lower() == 'nan':
            return self._get_empty_volume(modality), 0.0
        
        try:
            vol = np.load(path).astype(np.float32)
            # Crop använder ORIGINAL formen (mod_shape)
            shape = self.mod_shape[modality]
            
            if vol.ndim == 4: vol = vol[:, :, :, 1]
            
            vol = crop_fixed_rectangular(vol, target_shape=shape, threshold_percent=0.05)
            vol_tensor = torch.from_numpy(vol)
            
            if vol_tensor.ndim == 3: vol_tensor = vol_tensor.unsqueeze(0)
            elif vol_tensor.ndim == 4 and vol_tensor.shape[0] > 1: vol_tensor = vol_tensor[0:1, ...]

            return vol_tensor, 1.0 
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self._get_empty_volume(modality), 0.0

    def __getitem__(self, idx):
        label = int(self.y[idx])
        out_dict = {}
        masks = {}

        for mod in self.modalities:
            path = self.X_paths[mod][idx]
            volume, mask = self._load_volume(path, mod)
            
            if mask == 1.0 and self.transform:
                # --- ÄNDRING: Hämta modalitetsspecifik transform ---
                if isinstance(self.transform, dict):
                    mod_transform = self.transform[mod]
                    volume = mod_transform(volume)
                else:
                    # Fallback om man råkar skicka en vanlig transform
                    volume = self.transform(volume)

                mi, ma = volume.min(), volume.max()
                if ma > mi: volume = (volume - mi) / (ma - mi)
            
            out_dict[mod] = volume
            masks[mod] = torch.tensor([mask], dtype=torch.float32)

        return out_dict, masks, label


class TwoAugSelfSupervisedDataset(torch.utils.data.Dataset):
    # ÄNDRING: target_shapes istället för img_shape
    def __init__(self, X_paths, y, dic_classes, target_shapes, mod_shape, transform=None):
        self.X_paths = X_paths
        self.y = y
        self.classes = list(dic_classes.keys())
        self.transform = transform # Dictionary
        self.modalities = list(X_paths.keys())
        self.target_shapes = target_shapes
        self.mod_shape = mod_shape

    def __len__(self):
        return len(self.y)
    
    def _get_empty_volume(self, modality):
        t_shape = self.target_shapes[modality]
        return torch.zeros((3, *t_shape), dtype=torch.float32)

    def _process_view(self, volume, is_present, modality):
        """Applicerar modalitets-specifik transform."""
        if is_present and self.transform:
            # --- ÄNDRING: Välj rätt transform ---
            if isinstance(self.transform, dict):
                vol = self.transform[modality](volume)
            else:
                vol = self.transform(volume)
            
            mi, ma = vol.min(), vol.max()
            if ma > mi: vol = (vol - mi) / (ma - mi)
            return vol
        return volume

    def __getitem__(self, idx):
        label = int(self.y[idx])
        view1_dict = {}
        view2_dict = {}
        masks = {}

        for mod in self.modalities:
            path = self.X_paths[mod][idx]
            
            if pd.isna(path) or str(path).lower() == 'nan':
                raw_tensor = self._get_empty_volume(mod) 
                is_present = False
                mask_val = 0.0
            else:
                try:
                    raw_vol = np.load(path).astype(np.float32)
                    shape = self.mod_shape[mod]
                    if raw_vol.ndim == 4: raw_vol = raw_vol[:, :, :, 1]
                    vol = crop_fixed_rectangular(raw_vol, target_shape=shape, threshold_percent=0.05)
                    raw_tensor = torch.from_numpy(vol)
                    
                    if raw_tensor.ndim == 3: raw_tensor = raw_tensor.unsqueeze(0)
                    is_present = True
                    mask_val = 1.0
                except Exception as e:
                    print(f"[WARNING] Error loading {path}: {e}")
                    raw_tensor = self._get_empty_volume(mod)
                    is_present = False
                    mask_val = 0.0

            # Skicka med 'mod' så vi vet vilken transform som ska användas
            view1_dict[mod] = self._process_view(raw_tensor, is_present, mod)
            
            if is_present: raw_tensor_clone = raw_tensor.clone()
            else: raw_tensor_clone = raw_tensor
                
            view2_dict[mod] = self._process_view(raw_tensor_clone, is_present, mod)
            masks[mod] = torch.tensor([mask_val], dtype=torch.float32)

        return view1_dict, view2_dict, masks, label

def create_datasets(directory_dataframe, transforms_dic, dic_classes, n_fold, current_fold, test_split, seed, target_shapes, modalities, mod_shape):
    X_train, y_train, _ = get_mm_paths(directory_dataframe, dic_classes, "train", True, n_fold, current_fold, test_split, seed, modalities)
    X_val, y_val, _ = get_mm_paths(directory_dataframe, dic_classes, "val", False, n_fold, current_fold, test_split, seed, modalities)
    X_test, y_test, _ = get_mm_paths(directory_dataframe, dic_classes, "test", False, n_fold, current_fold, test_split, seed, modalities)

    # transforms_dic är nu en nestlad dict: transforms_dic['train']['mri'], transforms_dic['train']['amy'] etc.

    trainset = TwoAugSelfSupervisedDataset(X_train, y_train, dic_classes, target_shapes, mod_shape, transform=transforms_dic["train"])
    trainset_pretraining = TwoAugSelfSupervisedDataset(X_train, y_train, dic_classes, target_shapes, mod_shape, transform=transforms_dic["train"])
    trainset_normal = AugSupervisedDataset(X_train, y_train, dic_classes, target_shapes, mod_shape, transform=transforms_dic["train_noaug"])
    trainset_normal_augment = AugSupervisedDataset(X_train, y_train, dic_classes, target_shapes, mod_shape, transform=transforms_dic["train"])
    projectset = AugSupervisedDataset(X_train, y_train, dic_classes, target_shapes, mod_shape, transform=transforms_dic["project_noaug"])
    valset = AugSupervisedDataset(X_val, y_val, dic_classes, target_shapes, mod_shape, transform=transforms_dic["val"])
    testset = AugSupervisedDataset(X_test, y_test, dic_classes, target_shapes, mod_shape, transform=transforms_dic["test"])
    testset_projection = AugSupervisedDataset(X_test, y_test, dic_classes, target_shapes, mod_shape, transform=transforms_dic["test_projection"])

    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection


def get_brains(dataset_path, metadata_path, target_shapes, channels, dic_classes, n_fold, current_fold, test_split, seed, modalities, mod_shape):
    
    aug_prob = 0.5
    rand_rot = 6                        # random rotation range [deg]
    rand_rot_rad = rand_rot*math.pi/180 # random rotation range [rad]
    rand_noise_std = 0.01               # std random Gaussian noise
    rand_shift = 5                      # px random shift
    min_zoom = 0.9
    max_zoom = 1.1
    scale_dev = max_zoom - 1.0
    # rand_rot_rad = 6 * math.pi / 180
    
    # --- HJÄLPFUNKTION FÖR ATT SKAPA MODALITETS-SPECIFIKA TRANSFORMS ---
    def get_transform_chain(stage, target_shape):
        """
        Bygger en Compose-kedja för en specifik modalitet (baserat på dess target_shape).
        """
        if stage == 'train':
            return Compose([
                Resize(spatial_size=target_shape),
                # RandAffine(
                #     prob=aug_prob,
                #     rotate_range=(rand_rot_rad, rand_rot_rad, rand_rot_rad),
                #     translate_range=(rand_shift, rand_shift, rand_shift),
                #     scale_range=(scale_dev, scale_dev, scale_dev),
                #     mode='bilinear',       # Snabbare än bicubic
                #     padding_mode='zeros',  # Fyller tomrum med svart
                #     spatial_size=target_shape, 
                #     cache_grid=True # Nu fungerar cachingen korrekt!
                #     # cache_grid=True        # Snabbar upp beräkningen om input-storleken är konstant
                # ),
                # RandRotate(range_x=rand_rot_rad, range_y=rand_rot_rad, range_z=rand_rot_rad, prob=aug_prob),
                # # RandGaussianNoise(std=0.01, prob=aug_prob),
                # Affine(translate_params=(rand_shift, rand_shift, rand_shift), image_only=True),
                # RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=aug_prob),
                RepeatChannel(repeats=channels),
            ])
        else: # val, test, noaug
            return Compose([
                Resize(spatial_size=target_shape),
                RepeatChannel(repeats=channels),
            ])

    # --- BYGG TRANSFORMS DICTIONARY (Nested) ---
    # Structure: transforms_dic['train']['mri'] -> Compose(...)
    stages = ['train', 'train_noaug', 'project_noaug', 'val', 'test', 'test_projection']
    transforms_dic = {}

    for stage in stages:
        transforms_dic[stage] = {}
        for mod in modalities:
            # Hämta specifik output-storlek för denna modalitet
            t_shape = target_shapes[mod]
            
            # Avgör om det är 'train' (med aug) eller 'noaug'
            if stage == 'train' or stage == 'train_normal_augment':
                transforms_dic[stage][mod] = get_transform_chain('train', t_shape)
            else:
                transforms_dic[stage][mod] = get_transform_chain('other', t_shape)

    mm_df = load_npy_dataset(adni_path=dataset_path, classes=dic_classes.keys(), modalities=modalities)

    return create_datasets(
        directory_dataframe = mm_df,
        transforms_dic = transforms_dic,
        dic_classes = dic_classes,
        n_fold = n_fold,
        current_fold = current_fold,
        test_split = test_split,
        seed = seed,
        target_shapes = target_shapes, # Skickar med dict istället för tuple
        modalities = modalities,
        mod_shape=mod_shape
    )


def get_data(args: argparse.Namespace): 

    """ Load dataset based on the parsed arguments """

    mm_df = load_npy_dataset(adni_path=args.dataset_path, classes=list(args.dic_classes.keys()), modalities=args.modalities)

    # --- NYTT: Beräkna Target Shapes för VARJE modalitet separat ---
    ds = args.downscaling
    target_shapes = {} # Dict: {'mri': (42, 52, 44), 'amy': (32, 32, 32)}

    for mod in args.modalities:
        if mod not in args.mod_shape:
            print(f"[ERROR] Modalitet {mod} saknas i args.mod_shape config!", flush=True)
            continue
            
        orig_shape = args.mod_shape[mod] # T.ex. (169, 208, 179)
        
        # Beräkna nedskalad storlek
        new_slices = orig_shape[0] // ds
        new_rows = orig_shape[1] // ds
        new_cols = orig_shape[2] // ds
        
        target_shapes[mod] = (new_slices, new_rows, new_cols)
        
        print(f"[INFO] {mod.upper()} Orig: {orig_shape} -> Target (ds={ds}): {target_shapes[mod]}", flush=True)

    # För bakåtkompatibilitet sätter vi args.img_shape till MRIs shape (används kanske i main-scriptet för logging)
    if 'mri' in target_shapes:
        args.img_shape = target_shapes['mri']
        args.slices, args.rows, args.cols = args.img_shape
    else:
        # Fallback om man kör utan MRI
        first_mod = list(target_shapes.keys())[0]
        args.img_shape = target_shapes[first_mod]
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    return get_brains(
        dataset_path = args.dataset_path,
        metadata_path = args.metadata_path, 
        target_shapes = target_shapes, # NYTT
        channels = args.channels,
        dic_classes = args.dic_classes,
        n_fold = args.n_fold,
        current_fold = args.current_fold,
        test_split = args.test_split,
        seed = args.seed,
        modalities = args.modalities,
        mod_shape = args.mod_shape
        )

    raise Exception(f'Could not load data set, data set "{args.dataset_path}" not found!')
    

def get_dataloaders(args: argparse.Namespace):
    
    """ Get data loaders """
        
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection = get_data(args)
    
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = args.num_workers
    pretrain_batchsize = args.batch_size_pretrain 
    
    trainloader = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = args.batch_size,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = cuda,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(args.seed),
        drop_last = True)
           
    trainloader_pretraining = torch.utils.data.DataLoader(
        dataset = trainset_pretraining,
        batch_size = pretrain_batchsize,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = cuda,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(args.seed),
        drop_last = True)
    
    trainloader_normal = torch.utils.data.DataLoader(
        dataset = trainset_normal,
        batch_size = args.batch_size,
        shuffle = False, 
        sampler = sampler,
        pin_memory = cuda,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(args.seed),
        drop_last = True)
        
    trainloader_normal_augment = torch.utils.data.DataLoader(
        dataset = trainset_normal_augment,
        batch_size = args.batch_size,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = cuda,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(args.seed),
        drop_last = True)
    
    projectloader = torch.utils.data.DataLoader(
        dataset = projectset,
        batch_size = 1,
        shuffle = False, 
        sampler = sampler,
        pin_memory = cuda,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(args.seed),
        drop_last = True)
    
    valloader = torch.utils.data.DataLoader(
        dataset = valset,
        batch_size = 1,
        shuffle = True, 
        pin_memory = cuda,
        num_workers = num_workers,                
        worker_init_fn = np.random.seed(args.seed),
        drop_last = False)

    testloader = torch.utils.data.DataLoader(
        dataset = testset,
        batch_size = 1,
        shuffle = False, 
        pin_memory = cuda,
        num_workers = num_workers,                
        worker_init_fn = np.random.seed(args.seed),
        drop_last = False)
    
    test_projectloader = torch.utils.data.DataLoader(
        dataset = testset_projection,
        batch_size = 1,
        shuffle = False, 
        pin_memory = cuda,
        num_workers = num_workers,                
        worker_init_fn = np.random.seed(args.seed),
        drop_last = False)

    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, valloader, testloader, test_projectloader