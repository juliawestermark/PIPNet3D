#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:24:11 2023

@author: lisadesanti
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
    RandGaussianNoise,
    RandZoom,
    RepeatChannel,
)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from make_mri_dataset import setup_npy_mri_dataframe
    

def build_preprocessed_paths(df, preprocessed_root, dic_classes):
    """
    Bygger paths till .npy och omvandlar clinical_stage → label.
    Returnerar:
        X_paths : np.array med paths
        y       : np.array med int labels
    """
    paths = []
    labels = []

    for _, row in df.iterrows():
        npy_path = row["file_path"]

        paths.append(npy_path)
        labels.append(dic_classes[row["clinical_stage"]])

    return np.array(paths), np.array(labels)


def remove_subject_leakage(train_df, test_df):
    """Flyttar bort subjekt som hamnat i både train och test."""
    train_subj = set(train_df["individual_id"])
    test_subj = set(test_df["individual_id"])

    duplicates = train_subj.intersection(test_subj)

    for subj in duplicates:
        rows = test_df[test_df["individual_id"] == subj]
        train_df = pd.concat([train_df, rows], ignore_index=True)
        test_df = test_df.drop(rows.index)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def subject_kfold_split(df, labels, n_fold, current_fold):
    df = df.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_fold, shuffle=False)
    generator = skf.split(df, labels)

    for _ in range(current_fold):
        train_idx, val_idx = next(generator)

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True)
    )


def shuffle_arrays(X, y, seed=42):
    idx = np.arange(len(y))
    np.random.default_rng(seed).shuffle(idx)
    return X[idx], y[idx]


def get_mri_brains_paths(
        directory_dataframe,
        preprocessed_root,
        dic_classes={"CN":0,"MCI":1,"AD":2},
        set_type='train',
        shuffle=True,
        n_fold=5,
        current_fold=1,
        test_split=0.2,
        seed=42
    ):
    """
    ADNI MRI split + k-fold, helt utan nestlade funktioner och utan age.
    Returnerar:
        X_paths : np.array
        y       : np.array
        info    : dict
    """

    labels = directory_dataframe["clinical_stage"]

    # --- 1. subject-level train/val vs test ---
    X_train_val_df, X_test_df, y_train_val, y_test = train_test_split(
        directory_dataframe,
        labels,
        test_size=test_split,
        stratify=labels,
        random_state=seed,
        shuffle=True
    )

    # --- 2. ta bort subject leakage ---
    X_train_val_df, X_test_df = remove_subject_leakage(
        X_train_val_df, 
        X_test_df
    )
    y_test = X_test_df["clinical_stage"]

    # --- 3. k-fold split ---
    X_train_df, X_val_df = subject_kfold_split(
        X_train_val_df,
        X_train_val_df["clinical_stage"],
        n_fold=n_fold,
        current_fold=current_fold
    )

    # --- 4. bygg paths + labels ---
    X_train, y_train = build_preprocessed_paths(
        X_train_df, preprocessed_root, dic_classes
    )
    X_val, y_val = build_preprocessed_paths(
        X_val_df, preprocessed_root, dic_classes
    )
    X_test, y_test = build_preprocessed_paths(
        X_test_df, preprocessed_root, dic_classes
    )

    # --- 5. shuffle ---
    if shuffle:
        X_train, y_train = shuffle_arrays(X_train, y_train, seed)
        X_val, y_val     = shuffle_arrays(X_val,   y_val,   seed)
        X_test, y_test   = shuffle_arrays(X_test,  y_test,  seed)

    # --- 6. dataset info ---
    info = {
        "train_subjects": X_train_df["individual_id"].unique().tolist(),
        "val_subjects":   X_val_df["individual_id"].unique().tolist(),
        "test_subjects":  X_test_df["individual_id"].unique().tolist(),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    # print("Class distribution:")
    # print("Train set:")
    # print(X_train_df["clinical_stage"].value_counts())
    # print("Validation set:")
    # print(X_val_df["clinical_stage"].value_counts())
    # print("Test set:")
    # print(X_test_df["clinical_stage"].value_counts())

    # --- 7. returnera rätt split ---
    if set_type == "train":
        return X_train, y_train, info
    elif set_type == "val":
        return X_val, y_val, info
    elif set_type == "test":
        return X_test, y_test, info

    # fallback – returnera allt
    return build_preprocessed_paths(directory_dataframe, preprocessed_root, dic_classes)


class AugSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, X_paths, y, dic_classes, transform=None, mask=None):
        self.X_paths = X_paths
        self.y = y
        self.transform = transform
        self.img_dir = X_paths
        self.img_labels = y
        self.classes = list(dic_classes.keys())
        self.class_to_idx = dic_classes
        self.mask = mask

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        path = self.X_paths[idx]
        label = int(self.y[idx])

        # 1. Ladda data (D, H, W)
        img_arr = np.load(path).astype(np.float32)
        
        # 2. Hantera Mask
        if self.mask is not None:
            mask_arr = self.mask.copy() # (D, H, W)
        else:
            mask_arr = np.ones_like(img_arr)

        # 3. Slå ihop dem till (2, D, H, W)
        # Kanal 0 = Bild, Kanal 1 = Mask
        combined = np.stack([img_arr, mask_arr], axis=0)
        combined_tensor = torch.from_numpy(combined) 

        # 4. Applicera Transform (På båda kanalerna samtidigt!)
        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        
        # 5. Normalisering (OBS: Bara på bild-kanalen, inte masken!)
        img_part = combined_tensor[0, ...]
        mask_part = combined_tensor[1, ...] # Efter transform är masken roterad!

        mi = img_part.min()
        ma = img_part.max()
        if ma > mi:
            img_part = (img_part - mi) / (ma - mi)
        
        # Tröskla masken igen ifall interpolation gjorde den "suddig" i kanten
        mask_part = (mask_part > 0.5).float()
        
        # Applicera masken på input-bilden för säkerhets skull
        img_part = img_part * mask_part

        # 6. Returnera som (2, D, H, W) igen
        # Vi skickar masken "gömd" i datan till träningsloopen
        final_tensor = torch.stack([img_part, mask_part], dim=0)
        
        return final_tensor, label


class TwoAugSelfSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, X_paths, y, dic_classes, transform=None, mask=None):
        self.X_paths = X_paths
        self.y = y
        self.transform1 = transform
        self.transform2 = transform
        self.img_dir = X_paths
        self.img_labels = y
        self.classes = list(dic_classes.keys())
        self.class_to_idx = dic_classes
        self.mask = mask

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        path = self.X_paths[idx]
        label = int(self.y[idx])
        
        img_arr = np.load(path).astype(np.float32)
        if self.mask is not None:
            mask_arr = self.mask.copy()
        else:
            mask_arr = np.ones_like(img_arr)
            
        combined = np.stack([img_arr, mask_arr], axis=0)
        combined_tensor = torch.from_numpy(combined) # (2, D, H, W)

        # Transform 1
        if self.transform1:
            res1 = self.transform1(combined_tensor.clone())
            img1 = res1[0, ...]
            msk1 = (res1[1, ...] > 0.5).float()
            
            # Normalisera
            mi, ma = img1.min(), img1.max()
            if ma > mi: img1 = (img1 - mi)/(ma - mi)
            img1 = img1 * msk1
            out1 = torch.stack([img1, msk1], dim=0)
        else:
            # Hantera fall utan transform om nödvändigt...
            out1 = combined_tensor.clone()

        # Transform 2 (Gör samma sak...)
        if self.transform2:
            res2 = self.transform2(combined_tensor.clone())
            img2 = res2[0, ...]
            msk2 = (res2[1, ...] > 0.5).float()
            
            mi, ma = img2.min(), img2.max()
            if ma > mi: img2 = (img2 - mi)/(ma - mi)
            img2 = img2 * msk2
            out2 = torch.stack([img2, msk2], dim=0)
        else:
            out2 = combined_tensor.clone()

        return out1, out2, label


def create_datasets(
        directory_dataframe: pd.DataFrame,
        preprocessed_root: str,
        transforms_dic: dict,
        dic_classes = {"CN":0, "MCI":1, "AD":2},
        n_fold = 5,
        current_fold = 1,
        test_split = 0.2,
        seed = 42,
        mask = None
    ):
    """
    Skapar alla dataset:
        - trainset (self-supervised, 2 augment)
        - trainset_pretraining (self-supervised, 2 augment)
        - trainset_normal (supervised, no aug)
        - trainset_normal_augment (supervised, aug)
        - projectset (supervised noaug)
        - valset
        - testset
        - testset_projection
    """

    X_train, y_train, info_train = get_mri_brains_paths(
        directory_dataframe=directory_dataframe,
        preprocessed_root=preprocessed_root,
        dic_classes=dic_classes,
        set_type="train",
        shuffle=True,
        n_fold=n_fold,
        current_fold=current_fold,
        test_split=test_split,
        seed=seed
    )

    X_val, y_val, info_val = get_mri_brains_paths(
        directory_dataframe=directory_dataframe,
        preprocessed_root=preprocessed_root,
        dic_classes=dic_classes,
        set_type="val",
        shuffle=False,
        n_fold=n_fold,
        current_fold=current_fold,
        test_split=test_split,
        seed=seed
    )

    X_test, y_test, info_test = get_mri_brains_paths(
        directory_dataframe=directory_dataframe,
        preprocessed_root=preprocessed_root,
        dic_classes=dic_classes,
        set_type="test",
        shuffle=False,
        n_fold=n_fold,
        current_fold=current_fold,
        test_split=test_split,
        seed=seed
    )

    trainset = TwoAugSelfSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        transform=transforms_dic["train"],
        mask=mask
    )

    trainset_pretraining = TwoAugSelfSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        transform=transforms_dic["train"],
        mask=mask
    )

    trainset_normal = AugSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        transform=transforms_dic["train_noaug"],
        mask=mask
    )

    trainset_normal_augment = AugSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        transform=transforms_dic["train"],
        mask=mask
    )

    projectset = AugSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        transform=transforms_dic["project_noaug"],
        mask=mask
    )

    valset = AugSupervisedDataset(
        X_paths=X_val,
        y=y_val,
        dic_classes=dic_classes,
        transform=transforms_dic["val"],
        mask=mask
    )

    testset = AugSupervisedDataset(
        X_paths=X_test,
        y=y_test,
        dic_classes=dic_classes,
        transform=transforms_dic["test"],
        mask=mask
    )

    testset_projection = AugSupervisedDataset(
        X_paths=X_test,
        y=y_test,
        dic_classes=dic_classes,
        transform=transforms_dic["test_projection"],
        mask=mask
    )

    return (
        trainset,
        trainset_pretraining,
        trainset_normal,
        trainset_normal_augment,
        projectset,
        valset,
        testset,
        testset_projection
    )


def get_brains(
        dataset_path:str,
        metadata_path: str,
        img_shape: tuple,
        channels: int,
        dic_classes = {'CN':0,'MCI':1,'AD':2},
        n_fold = 5,
        current_fold = 1,
        test_split = 0.2,
        seed = 42,
        mask = None):
    
    # Data augmentation (on-the-fly) parameters
    aug_prob = 0.5
    rand_rot = 6                        # random rotation range [deg]
    rand_rot_rad = rand_rot*math.pi/180 # random rotation range [rad]
    rand_noise_std = 0.01               # std random Gaussian noise
    rand_shift = 5                      # px random shift
    min_zoom = 0.9
    max_zoom = 1.1
    
    transforms_dic = {
        'train': Compose([
            Resize(spatial_size=img_shape),
            RandRotate(range_x=rand_rot_rad, range_y=rand_rot_rad, range_z=rand_rot_rad, prob=aug_prob),
            RandGaussianNoise(std=rand_noise_std, prob=aug_prob),
            Affine(translate_params=(rand_shift, rand_shift, rand_shift), image_only=True),
            RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=aug_prob),
            RepeatChannel(repeats=channels),
        ]),
        'train_noaug': Compose([
            Resize(spatial_size = img_shape),
            RepeatChannel(repeats=channels),
        ]),
        'project_noaug': Compose([
            Resize(spatial_size = img_shape),
            RepeatChannel(repeats=channels),
        ]),
        'val': Compose([
            Resize(spatial_size = img_shape),
            RepeatChannel(repeats=channels),
        ]),
        'test': Compose([
            Resize(spatial_size = img_shape),
            RepeatChannel(repeats=channels),
        ]),
        'test_projection': Compose([
            Resize(spatial_size = img_shape),
            RepeatChannel(repeats=channels),
        ]),
    }

    directory_dataframe = setup_npy_mri_dataframe(classes=dic_classes.keys(), adni_path=dataset_path)

    return create_datasets(
        directory_dataframe = directory_dataframe,
        preprocessed_root = dataset_path,
        transforms_dic = transforms_dic,
        dic_classes = dic_classes,
        n_fold = n_fold,
        current_fold = current_fold,
        test_split = test_split,
        seed = seed,
        mask = mask
    )


def get_data(args: argparse.Namespace): 

    """ Load dataset based on the parsed arguments """

    # --- 1. Ladda masken ---
    mask_path = args.global_mask_path
    mask = None
    
    if os.path.exists(mask_path):
        print(f"[INFO] Loading global mask from {mask_path}")
        mask = np.load(mask_path).astype(np.float32)
        
        # Säkerhetskontroll: Masken måste vara 1 där vi vill behålla data, 0 annars.
        # Om den är bool (True/False), konvertera.
        if mask.dtype == bool:
            mask = mask.astype(np.float32)
    else:
        print(f"[WARNING] No global mask found at {mask_path}. Training without mask.")

    # -----------------------

    sample_df = setup_npy_mri_dataframe(
        classes=args.dic_classes.keys(),
        adni_path=args.dataset_path
    )

    sample_path = sample_df["file_path"].iloc[0]
    sample_img = np.load(sample_path)  # shape: (D, H, W)

    orig_slices, orig_rows, orig_cols = sample_img.shape

    ds = args.downscaling
    args.slices = orig_slices // ds
    args.rows   = orig_rows   // ds
    args.cols   = orig_cols   // ds
    args.img_shape = (args.slices, args.rows, args.cols)

    print(f"[INFO] Image shape detected from data: {sample_img.shape}")
    print(f"[INFO] Using downscaled shape: {args.img_shape}")

    # Check dimensions
    sample_path = sample_df["file_path"].iloc[0]
    sample_img = np.load(sample_path)
    if mask is not None:
        if mask.shape != sample_img.shape:
            print(f"[ERROR] Mask shape {mask.shape} does not match image shape {sample_img.shape}")
            # Här kan du välja att crasha eller försöka resizea masken.
            # För säkerhets skull rekommenderar jag att du genererar masken på rätt data.
            raise ValueError("Mask dimensions mismatch!")
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    return get_brains(
        dataset_path = args.dataset_path,
        metadata_path = args.metadata_path, 
        img_shape = args.img_shape,
        channels = args.channels,
        dic_classes = args.dic_classes,
        n_fold = args.n_fold,
        current_fold = args.current_fold,
        test_split = args.test_split,
        seed = args.seed,
        mask = mask
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

