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

from make_mm_dataset import load_npy_dataset

def print_split_statistics(df, split_name, modalities):
    """
    Dynamisk statistikräknare för godtyckligt antal modaliteter.
    
    Args:
        df: DataFramen för den aktuella spliten.
        split_name: Namn på spliten (t.ex. "Train", "Val").
        modalities: Lista med strängar, t.ex. ['mri', 'amy', 'tau'].
    """
    total = len(df)
    print(f"\n{'='*10} Statistics for {split_name.upper()} Set {'='*10}")
    print(f"Total entries (rows): {total}")
    
    # Håll koll på kolumner vi faktiskt hittade
    found_cols = []
    
    # 1. Statistik per modalitet
    for mod in modalities:
        col_name = f"file_path_{mod}"
        
        if col_name in df.columns:
            # Räkna rader som INTE är NaN och INTE är tomma strängar
            count = df[col_name].apply(lambda x: pd.notna(x) and str(x).strip() != "").sum()
            percent = (count / total * 100) if total > 0 else 0
            
            print(f"  - Has {mod.upper().ljust(5)}: {count} ({percent:.1f}%)")
            found_cols.append(col_name)
        else:
            print(f"  - Has {mod.upper().ljust(5)}: 0 (Column '{col_name}' missing)")

    # 2. Statistik för snittet (De som har ALLA modaliteter)
    if found_cols:
        # Skapa en mask som är True från början
        has_all_mask = pd.Series([True] * total, index=df.index)
        
        for col in found_cols:
            # Uppdatera masken: Måste ha denna modalitet OCH tidigare modaliteter
            is_valid = df[col].apply(lambda x: pd.notna(x) and str(x).strip() != "")
            has_all_mask = has_all_mask & is_valid
            
        n_complete = has_all_mask.sum()
        pct_complete = (n_complete / total * 100) if total > 0 else 0
        
        print(f"  ------------------------")
        print(f"  - Complete Intersection ({'+'.join(modalities)}): {n_complete} ({pct_complete:.1f}%)")
    
    print("="*40)


def build_preprocessed_paths(df, dic_classes, modalities):
    """
    Extraherar paths dynamiskt för alla angivna modaliteter.
    Returnerar:
        X_paths : Dict {modality: np.array}
        y       : np.array med int labels
    """
    # 1. Initiera dictionary med tomma listor för varje modalitet
    # Ex: {'mri': [], 'amy': []}
    paths_dict = {mod: [] for mod in modalities}
    labels = []

    for _, row in df.iterrows():
        labels.append(dic_classes[row["clinical_stage"]])

        # 2. Loopa över modaliteterna för att hitta rätt kolumn
        for mod in modalities:
            # Konstruera kolumnnamnet dynamiskt
            col_name = f"file_path_{mod}" 
            
            # Hämta path (använd .get för säkerhet om kolumnen saknas tillfälligt)
            # Detta lägger till path eller NaN/None i listan
            path = row.get(col_name, pd.NA)
            paths_dict[mod].append(path)
            # print(f"COL_NAME: {col_name}")
            # print(f"PATH: {path}")

    # 3. Konvertera listorna till numpy arrays och paketera i X_paths
    X_paths = {mod: np.array(paths) for mod, paths in paths_dict.items()}
    
    return X_paths, np.array(labels)


def shuffle_arrays(X_dict, y, seed=42):
    """Shufflar alla modaliteter i synk med labels"""
    # Ta längden från första modaliteten
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

    # 2. Split: Train+Val vs Test
    train_val_subj, test_subj, train_val_labels, _ = train_test_split(
        unique_subjects,
        unique_labels,
        test_size=test_split,
        stratify=unique_labels,
        random_state=seed,
        shuffle=True
    )

    # 3. K-Fold Split: Train vs Val
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    split_generator = skf.split(train_val_subj, train_val_labels)
    
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(split_generator):
        if i == (current_fold - 1):
            train_idx = t_idx
            val_idx = v_idx
            break
            
    if train_idx is None:
        raise ValueError(f"Invalid fold {current_fold} for n_fold={n_fold}")

    final_train_subj = train_val_subj[train_idx]
    final_val_subj   = train_val_subj[val_idx]

    # 4. Filtrera original-dataframen
    X_train_df = directory_dataframe[directory_dataframe["individual_id"].isin(final_train_subj)].reset_index(drop=True)
    X_val_df   = directory_dataframe[directory_dataframe["individual_id"].isin(final_val_subj)].reset_index(drop=True)
    X_test_df  = directory_dataframe[directory_dataframe["individual_id"].isin(test_subj)].reset_index(drop=True)

    # 5. Bygg paths dicts
    X_train, y_train = build_preprocessed_paths(X_train_df, dic_classes, modalities)
    X_val, y_val     = build_preprocessed_paths(X_val_df, dic_classes, modalities)
    X_test, y_test   = build_preprocessed_paths(X_test_df, dic_classes, modalities)

    # 6. Shuffle
    if shuffle:
        X_train, y_train = shuffle_arrays(X_train, y_train, seed)
        X_val, y_val     = shuffle_arrays(X_val,   y_val,   seed)
        X_test, y_test   = shuffle_arrays(X_test,  y_test,  seed)

    # 7. Info
    info = {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }
    
    print(f"Fold {current_fold} ({set_type}): {len(y_train)} train, {len(y_val)} val, {len(y_test)} test images.")

    if set_type == "train":
        print_split_statistics(X_train_df, set_type, modalities)
        return X_train, y_train, info
    elif set_type == "val":
        print_split_statistics(X_val_df, set_type, modalities)
        return X_val, y_val, info
    elif set_type == "test":
        print_split_statistics(X_test_df, set_type, modalities)
        return X_test, y_test, info
    
    
    return X_train, y_train, info


class AugSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, X_paths, y, dic_classes, img_shape, transform=None):
        self.X_paths = X_paths
        self.y = y
        self.transform = transform
        self.img_dir = X_paths
        self.img_labels = y
        self.classes = list(dic_classes.keys())
        self.class_to_idx = dic_classes
        self.modalities = list(X_paths.keys())
        self.img_shape = img_shape

    def __len__(self):
        return len(self.y)

    def _get_empty_volume(self, modality):
        # ÄNDRING: Returnera alltid 1 kanal. 
        # (Vi struntar i om det är 'amy' eller 'mri', vi vill ha konsekvens).
        return torch.zeros((3, *self.img_shape), dtype=torch.float32)

    def _load_volume(self, path, modality):
        # Om path är NaN (saknas)
        if pd.isna(path) or str(path).lower() == 'nan':
            return self._get_empty_volume(modality), 0.0
        
        try:
            vol = np.load(path).astype(np.float32)
            
            # --- NY LOGIK FÖR ATT TA BARA FÖRSTA KANALEN ---
            
            # Fall 1: Bilden är 4D (D, H, W, C) - Vanligt format om man sparar nifti till npy
            if vol.ndim == 4:
                # Vi tar bara ut index 0 från sista dimensionen (kanalen)
                # Resultatet blir 3D: (D, H, W)
                vol = vol[:, :, :, 0]
            
            # Fall 2: Bilden är redan 3D (D, H, W) - Gör ingenting
            
            # --- SLUT PÅ NY LOGIK ---

            # Nu konverterar vi till Tensor och lägger till kanal-dimensionen först
            # Resultat: (1, D, H, W)
            vol_tensor = torch.from_numpy(vol)
            
            # Säkerhetscheck ifall volymen råkade vara (C, D, H, W) från början
            if vol_tensor.ndim == 3:
                vol_tensor = vol_tensor.unsqueeze(0)
            elif vol_tensor.ndim == 4 and vol_tensor.shape[0] > 1:
                 # Om den var (C, D, H, W) tar vi första där också
                 vol_tensor = vol_tensor[0:1, ...]

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
            
            # #Dubbelkolla shape här innan return (DEBUG)
            # if volume.shape[0] != 1:
            #     print(f"SHAPE ERROR in {mod}: {volume.shape}")

            if mask == 1.0 and self.transform:
                volume = self.transform(volume)
                mi, ma = volume.min(), volume.max()
                if ma > mi:
                    volume = (volume - mi) / (ma - mi)
            
            out_dict[mod] = volume
            masks[mod] = torch.tensor([mask], dtype=torch.float32)

        return out_dict, masks, label


class TwoAugSelfSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, X_paths, y, dic_classes, img_shape, transform=None):
        self.X_paths = X_paths
        self.y = y
        self.img_dir = X_paths
        self.img_labels = y
        self.classes = list(dic_classes.keys())
        self.class_to_idx = dic_classes
        self.transform = transform
        self.modalities = list(X_paths.keys())
        self.img_shape = img_shape
        self.debug_counter = 0

    def __len__(self):
        return len(self.y)
    
    # --- FIX 1: Rätt kanaler vid tom data ---
    def _get_empty_volume(self, modality):
        # --- FIX: Returnera ALLTID 1 kanal (1, D, H, W) ---
        # Detta garanterar att vi kan stacka tensors även om en modalitet saknas.
        return torch.zeros((3, *self.img_shape), dtype=torch.float32)

    def _process_view(self, volume, is_present):
        """Applicerar transform och normalisering om bilden finns."""
        if is_present and self.transform:
            # Applicera slumpmässig augmentering
            vol = self.transform(volume)
            
            # Normalisera till 0-1 (Min-Max scaling)
            mi, ma = vol.min(), vol.max()
            if ma > mi:
                vol = (vol - mi) / (ma - mi)
            return vol
        
        # Om bilden inte finns (är noll-tensor) returnera den bara
        return volume

    def __getitem__(self, idx):
        label = int(self.y[idx])
        
        view1_dict = {}
        view2_dict = {}
        masks = {}

        for mod in self.modalities:
            path = self.X_paths[mod][idx]
            
            # 1. Hantera NaN / Saknad fil
            if pd.isna(path) or str(path).lower() == 'nan':
                raw_tensor = self._get_empty_volume(mod) 
                is_present = False
                mask_val = 0.0
            else:
                try:
                    # Ladda fil
                    raw_vol = np.load(path).astype(np.float32)
                    
                    # --- FIX: Hantera dimensioner (Tvinga till 1 kanal) ---
                    
                    # Om 4D (D, H, W, C), ta bara andra kanalen
                    if raw_vol.ndim == 4:
                        raw_vol = raw_vol[:, :, :, 1]
                    
                    # Konvertera till Tensor
                    raw_tensor = torch.from_numpy(raw_vol)
                    
                    # Lägg till kanal-dimension om den saknas: (D, H, W) -> (1, D, H, W)
                    if raw_tensor.ndim == 3:
                        raw_tensor = raw_tensor.unsqueeze(0)
                    # # Säkerhetscheck om den fortfarande är 4D (t.ex. C, D, H, W)
                    # elif raw_tensor.ndim == 4 and raw_tensor.shape[0] > 1:
                    #     raw_tensor = raw_tensor[0:1, ...]
                    
                    is_present = True
                    mask_val = 1.0
                    
                except Exception as e:
                    # Om filen är korrupt eller inte hittas
                    print(f"[WARNING] Error loading {path}: {e}")
                    raw_tensor = self._get_empty_volume(mod)
                    is_present = False
                    mask_val = 0.0

            # 2. Skapa två olika vyer (augmenteringar) för Self-Supervised Learning
            # Eftersom self.transform innehåller slumpmässighet (noise, rotation etc),
            # kommer view1 och view2 bli lite olika, vilket är poängen.
            
            # Vy 1
            view1_dict[mod] = self._process_view(raw_tensor, is_present)
            
            # Vy 2 (Vi klonar för säkerhets skull, även om transforms oftast inte sker in-place)
            if is_present:
                raw_tensor_clone = raw_tensor.clone()
            else:
                raw_tensor_clone = raw_tensor
                
            view2_dict[mod] = self._process_view(raw_tensor_clone, is_present)
            
            # Mask (samma för båda vyer)
            masks[mod] = torch.tensor([mask_val], dtype=torch.float32)

        return view1_dict, view2_dict, masks, label

def create_datasets(directory_dataframe, transforms_dic, dic_classes, n_fold, current_fold, test_split, seed, img_shape, modalities):
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

    # Hämta paths för alla set
    X_train, y_train, _ = get_mm_paths(
        directory_dataframe, dic_classes, "train", True, n_fold, current_fold, test_split, seed, modalities
    )
    X_val, y_val, _ = get_mm_paths(
        directory_dataframe, dic_classes, "val", False, n_fold, current_fold, test_split, seed, modalities
    )
    X_test, y_test, _ = get_mm_paths(
        directory_dataframe, dic_classes, "test", False, n_fold, current_fold, test_split, seed, modalities
    )

    trainset = TwoAugSelfSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["train"]
    )

    trainset_pretraining = TwoAugSelfSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["train"]
    )

    trainset_normal = AugSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["train_noaug"]
    )

    trainset_normal_augment = AugSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["train"]
    )

    projectset = AugSupervisedDataset(
        X_paths=X_train,
        y=y_train,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["project_noaug"]
    )

    valset = AugSupervisedDataset(
        X_paths=X_val,
        y=y_val,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["val"]
    )

    testset = AugSupervisedDataset(
        X_paths=X_test,
        y=y_test,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["test"]
    )

    testset_projection = AugSupervisedDataset(
        X_paths=X_test,
        y=y_test,
        dic_classes=dic_classes,
        img_shape=img_shape,
        transform=transforms_dic["test_projection"]
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
        modalities = ["mri"]):
    
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
            # RandRotate(range_x=rand_rot_rad, range_y=rand_rot_rad, range_z=rand_rot_rad, prob=aug_prob),
            RandGaussianNoise(std=rand_noise_std, prob=aug_prob),
            # Affine(translate_params=(rand_shift, rand_shift, rand_shift), image_only=True),
            # RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=aug_prob),
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

    mm_df = load_npy_dataset(adni_path=dataset_path, classes=dic_classes.keys(), modalities=modalities)

    return create_datasets(
        directory_dataframe = mm_df,
        transforms_dic = transforms_dic,
        dic_classes = dic_classes,
        n_fold = n_fold,
        current_fold = current_fold,
        test_split = test_split,
        seed = seed,
        img_shape = img_shape,
        modalities = modalities
    )


def get_data(args: argparse.Namespace): 

    """ Load dataset based on the parsed arguments """

    # 1. Ladda Master DataFrame (Multimodal)
    mm_df = load_npy_dataset(adni_path=args.dataset_path, classes=list(args.dic_classes.keys()), modalities=args.modalities)

    # 2. Check shapes dynamically
    reference_shape = None

    for mod in args.modalities:
        col_name = f"file_path_{mod}"
        
        # Hämta första raden som faktiskt har en fil (inte NaN) för denna modalitet
        # Detta är viktigt om datasetet är "sparse" (t.ex. rad 0 har MRI men saknar Amyloid)
        valid_rows = mm_df[mm_df[col_name].notna()]
        
        if valid_rows.empty:
            print(f"[WARNING] No valid files found for modality '{mod}' in the dataframe!", flush=True)
            continue
            
        # Ta första giltiga path
        sample_path = valid_rows.iloc[0][col_name]
        
        try:
            vol = np.load(sample_path)
            print(f"[INFO] {mod.upper()} Shape: {vol.shape}", flush=True)
            
            # Vi använder den första fungerande modaliteten vi hittar som "referens" för args.img_shape
            if reference_shape is None:
                reference_shape = vol.shape
                
        except Exception as e:
            print(f"[ERROR] Failed to load sample for {mod} at {sample_path}: {e}", flush=True)

    if reference_shape is None:
        raise ValueError("Could not determine image shape. Check paths and data availability.")

    # Beräkna downscaling baserat på referens-formen
    ds = args.downscaling
    orig_slices, orig_rows, orig_cols = reference_shape
    
    args.slices = orig_slices // ds
    args.rows   = orig_rows   // ds
    args.cols   = orig_cols   // ds
    args.img_shape = (args.slices, args.rows, args.cols)
    
    print(f"[INFO] Target Shape (downscaled by {ds}): {args.img_shape}", flush=True)
    
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
        modalities = args.modalities
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

