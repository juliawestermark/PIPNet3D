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
import pandas 
import nibabel as nib

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
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity
)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

ADNI_PATH = "/home/maia-user/ADNI_complete"
DATA_PATH = os.path.join(ADNI_PATH, "adni")
COLLECTION_PATH = os.path.join(ADNI_PATH, "OutputCollection.csv")
DEMOGRAPHICS_PATH = os.path.join(ADNI_PATH, "participant_demographics.csv")
DXSUM_PATH = os.path.join(ADNI_PATH, "DXSUM_PDXCONV_ADNIALL.csv")
MRI_FILENAME = "mri/rescaled_align_norm.nii.gz"

v_map = {
    "v02": 0,
    "v04": 3,
    "v05": 6,
    "v06": 6,
    "v11": 12*1,
    "v21": 12*2,
    "v31": 12*3,
    "v41": 12*4,
    "v51": 12*5,
}


def build_file_path(row: pd.Series) -> str:
    """Build the file path for a given row in the dataframe."""
    guid = row["Output collection GUID"]
    file_path = os.path.join(DATA_PATH, guid, MRI_FILENAME)
    return file_path


def determine_clinical_stage(row: pd.Series) -> str:
    """Determine the clinical stage based on diagnosis variables."""
    # Extract diagnosis variables
    dxcur = row.get("DXCURREN", pd.NA)
    dxchange = row.get("DXCHANGE", pd.NA)
    diagnosis = row.get("DIAGNOSIS", pd.NA)
    
    # Determine clinical stage based on ADNI3 (DIAGNOSIS)
    if pd.notna(diagnosis):
        if diagnosis == 1:
            return "CN"  # Cognitively Normal
        elif diagnosis == 2:
            return "MCI"  # Mild Cognitive Impairment
        elif diagnosis == 3:
            return "AD"  # Alzheimer's Disease
        
    # Determine clinical stage based on ADNIGO/2 (DXCHANGE)
    if pd.notna(dxchange):
        if dxchange in [1, 7, 9]:
            return "CN"  # Stable/Returned to Normal
        elif dxchange in [2, 8]:
            return "MCI"  # Stable/Returned to MCI
        elif dxchange in [3, 5, 6]:
            return "AD"  # Stable/Converted to Dementia
        elif dxchange == 4:
            return "MCI"  # Conversion to MCI
        
    # Determine clinical stage based on ADNI1 (DXCURREN)
    if pd.notna(dxcur):
        if dxcur == 1:
            return "CN"  # Cognitively Normal
        elif dxcur == 2:
            return "MCI"  # Mild Cognitive Impairment
        elif dxcur == 3:
            return "AD"  # Alzheimer's Disease

    return pd.NA  # If no valid data


def _get_first_entry(subject_data, viscode: str, column: str):
    entries = subject_data[subject_data["VISCODE2"] == viscode]
    if entries.empty:
        return pd.NA
    if len(entries) > 1:
        logger.warning("Multiple %s entries for subject %s", viscode, subject_data["individual_id"].iloc[0])
    return entries.iloc[0][column]


def get_baseline_date_for_subject(subject_id: str, dxsum: pd.DataFrame) -> pd.Timestamp:
    """Get the baseline date for a given subject ID."""
    subject_data = dxsum[dxsum["individual_id"] == subject_id]
    if subject_data.empty:
        logger.warning(f"No data found for subject {subject_id}.")
        return pd.NaT
    
    date = _get_first_entry(subject_data, "bl", "EXAMDATE")
    if pd.isna(date):
        date = _get_first_entry(subject_data, "sc", "EXAMDATE")

    if pd.isna(date):
        logger.warning(f"No valid baseline/screening date found for subject {subject_id}.")
        return pd.NaT

    return date

def get_baseline_class_for_subject(subject_id: str, dxsum: pd.DataFrame) -> str:
    """Get the baseline clinical stage for a given subject ID."""
    subject_data = dxsum[dxsum["individual_id"] == subject_id]
    if subject_data.empty:
        logger.warning(f"No data found for subject {subject_id}.")
        return "Unknown"
    
    stage = _get_first_entry(subject_data, "bl", "clinical_stage")
    if pd.isna(stage):
        stage = _get_first_entry(subject_data, "sc", "clinical_stage")

    if pd.isna(stage):
        logger.warning(f"No baseline/screening clinical stage for subject {subject_id}.")
        return "Unknown"
    
    return stage


def get_months_from_baseline(timepoint: str) -> float:
    """Convert timepoint string to months from baseline."""

    if pd.isna(timepoint):
        return pd.NA
    if timepoint in ["sc", "init"]:
        return 0
    if timepoint == "bl":
        return 0
    if timepoint.startswith("m"):
        try:
            return int(timepoint[1:])
        except ValueError:
            return pd.NA
    if timepoint.startswith("y"):
        try:
            years = int(timepoint[1:])
            return years * 12
        except ValueError:
            return pd.NA
    if timepoint.startswith("v"):
        if timepoint in v_map:
            return v_map[timepoint]
    if timepoint in ["tau"]:
        return pd.NA
    return pd.NA


def get_session_date(row: pd.Series) -> pd.Timestamp:
    """Calculate the session date based on baseline date and months from baseline."""
    baseline_date = row["baseline_date"]
    months_from_bl = row["months_from_baseline"]
    if pd.isna(baseline_date) or pd.isna(months_from_bl):
        logger.warning(f"Missing data for row {row.name} for subject {row['individual_id']}: baseline_date={baseline_date}, months_from_baseline={months_from_bl}")
        # detta är de som är tau istället för tidsperiod
        return pd.NA
    try:
        session_date = baseline_date + pd.DateOffset(months=months_from_bl)
        return session_date
    except Exception as e:
        logger.warning(f"Error calculating session date for row {row.name}: {e}")
        return pd.NA


def setup_mri_dataframe():

    # -- Load csv --
    csv_output_collection = pd.read_csv(COLLECTION_PATH)
    csv_dxsum = pd.read_csv(DXSUM_PATH)

    # -- Build filepath and inclusion flags --
    csv_output_collection["file_path"] = csv_output_collection.apply(build_file_path, axis=1)
    csv_output_collection["included"] = (
        (csv_output_collection["Job status"] == "completed")
        & csv_output_collection["file_path"].apply(os.path.exists)
        & (csv_output_collection["TimePoint"] != "tau")
        & csv_output_collection["Individual's ID"].isin(csv_dxsum["PTID"])
    )

    # -- Prepare mri dataset --
    mri = csv_output_collection[csv_output_collection["included"]].copy()
    mri = mri[[
        "Output collection GUID", 
        "Individual's ID", 
        "TimePoint", 
        "file_path"
    ]].rename(columns={
        "Output collection GUID": "exam_id",
        "Individual's ID": "individual_id",
        "TimePoint": "time_point"
    })

    # -- Prepare diagnosis summary --
    dxsum = csv_dxsum[[
        "PTID",
        "EXAMDATE",
        "USERDATE",
        "VISCODE", 
        "VISCODE2",
        "DXCURREN",
        "DXCHANGE",
        "DIAGNOSIS"
    ]].rename(columns={
        "PTID": "individual_id",
    })
    dxsum["EXAMDATE"] = dxsum["EXAMDATE"].fillna(dxsum["USERDATE"])
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"], errors='coerce')
    dxsum["clinical_stage"] = dxsum.apply(determine_clinical_stage, axis=1)
    dxsum = dxsum.dropna(subset=["clinical_stage"])

    # -- Add baseline info --
    mri["baseline_date"] = mri["individual_id"].apply(get_baseline_date_for_subject, args=(dxsum,))
    mri["baseline_class"] = mri["individual_id"].apply(get_baseline_class_for_subject, args=(dxsum,))
    mri["months_from_baseline"] = mri["time_point"].apply(get_months_from_baseline)
    mri["est_exam_date"] = mri.apply(get_session_date, axis=1)

    # -- Merge mri with diagnosis info --
    matched = pd.merge_asof(
        mri.dropna(subset=["est_exam_date"]).sort_values("est_exam_date"),
        dxsum[["individual_id", "EXAMDATE", "VISCODE2", "clinical_stage"]]
            .sort_values("EXAMDATE"),
        by="individual_id",
        left_on="est_exam_date",
        right_on="EXAMDATE",
        direction="nearest"   
    ).dropna(subset=["clinical_stage"])

    mri_merged = matched[[
        "exam_id", 
        "individual_id", 
        "EXAMDATE",
        "baseline_class",
        "baseline_date",
        "clinical_stage",
        "file_path",
    ]].rename(columns={
        "EXAMDATE": "exam_date"
    })

    return mri_merged

def get_small_mri_dataframe():
    mri = setup_mri_dataframe()
    small_mri = mri[[
        "individual_id",
        "clinical_stage",
        "file_path",
    ]].copy()
    return small_mri

def split_dataset(dataset: pd.DataFrame, test_size, val_size, random_state=42) -> tuple:
    """Splits the dataset into training, validation and testing sets based on unique subjects."""
    unique_subjects = dataset["individual_id"].unique()
    train_val_subjects, test_subjects = train_test_split(unique_subjects, test_size=test_size, random_state=random_state)
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=(val_size / (1-test_size)), random_state=random_state)  # 0.25 x 0.8 = 0.2
    train_df = dataset[dataset["individual_id"].isin(train_subjects)].copy()
    val_df = dataset[dataset["individual_id"].isin(val_subjects)].copy()
    test_df = dataset[dataset["individual_id"].isin(test_subjects)].copy()
    return train_df, val_df, test_df


class AugSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataframe,
                 transform=None,
                 label_map = {"CN": 0, "MCI": 1, "AD": 2}
                 ):
        
        self.dataframe = dataframe
        self.transform = transform
        # self.label_map = label_map
        self.img_dir = dataframe["file_path"]
        self.img_labels = dataframe["clinical_stage"]
        self.subjects = dataframe["individual_id"]
        self.class_to_idx = label_map

        # MONAI-transform pipeline
        self.monai_transform = Compose([
            LoadImage(image_only=True),    # laddar NIfTI som np.array
            EnsureChannelFirst(),          # lägger till kanal som första dimension (C, D, H, W)
            ScaleIntensity(),              # normaliserar intensiteten till [0,1]
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]["file_path"]
        # image = nib.load(img_path).get_fdata()
        # image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = self.monai_transform(img_path)  # direkt tensor-liknande np.array

        label = self.class_to_idx[self.dataframe.iloc[idx]["clinical_stage"]]

        if self.transform:
            image = self.transform(image)
            # img_min = image.min()
            # img_max = image.max()
            # image = (image-img_min)/(img_max-img_min)
        
        # konvertera till tensor om transform inte redan gör det
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        return image, label


class TwoAugSelfSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataframe,
                 transform=None,
                 label_map = {"CN": 0, "MCI": 1, "AD": 2}
                 ):
        
        self.dataframe = dataframe
        self.transform1 = transform
        self.transform2 = transform
        # self.label_map = label_map
        self.img_dir = dataframe["file_path"]
        self.img_labels = dataframe["clinical_stage"]
        self.subjects = dataframe["individual_id"]
        self.class_to_idx = label_map

        # MONAI-transform pipeline
        self.monai_transform = Compose([
            LoadImage(image_only=True),    # laddar NIfTI som np.array
            EnsureChannelFirst(),          # lägger till kanal som första dimension (C, D, H, W)
            ScaleIntensity(),              # normaliserar intensiteten till [0,1]
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]["file_path"]
        # image = nib.load(img_path).get_fdata()
        # image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = self.monai_transform(img_path)  # direkt tensor-liknande np.array

        label = self.class_to_idx[self.dataframe.iloc[idx]["clinical_stage"]]

        if self.transform1:
            image1 = self.transform1(image)
            # img_min = image1.min()
            # img_max = image1.max()
            # image1 = (image1-img_min)/(img_max-img_min)
        if self.transform2:
            image2 = self.transform2(image)
            # img_min = image2.min()
            # img_max = image2.max()
            # image2 = (image2-img_min)/(img_max-img_min)

        # konvertera till tensor om transform inte redan gör det
        if not isinstance(image1, torch.Tensor):
            image1 = torch.tensor(image1, dtype=torch.float32)
        # konvertera till tensor om transform inte redan gör det
        if not isinstance(image2, torch.Tensor):
            image2 = torch.tensor(image2, dtype=torch.float32)

        return image1, image2, label


def create_datasets(
        dataset_path: str,
        metadata_path: str,
        transforms_dic = None,
        dic_classes = {'CN':0,'MCI':1,'AD':2},
        n_fold = 5,
        current_fold = 1,
        test_split = 0.2,
        seed = 42):
    """ 
    Instantiates all the Dataset classes:
        - trainset
        - trainset_pretraining
        - trainset_normal
        - trainset_normal_augment
        - projectset
        - valset
        - testset
        - testset_projection """
    
    df = setup_mri_dataframe()
    train_df, val_df, test_df = split_dataset(df, test_size=test_split, val_size=test_split, random_state=seed)

    trainset = TwoAugSelfSupervisedDataset(
        train_df,
        transform=transforms_dic["train"],
        label_map = dic_classes
        )
    
    trainset_pretraining = TwoAugSelfSupervisedDataset(
        train_df,
        transform=transforms_dic["train"],
        label_map = dic_classes
        )
    
    trainset_normal = AugSupervisedDataset(
        train_df,
        transform=transforms_dic["train_noaug"],
        label_map = dic_classes
        )

    trainset_normal_augment = AugSupervisedDataset(
        train_df,
        transform=transforms_dic["train"],
        label_map = dic_classes
        )
    
    projectset = AugSupervisedDataset(
        train_df,
        transform=transforms_dic["project_noaug"],
        label_map = dic_classes
        )
    
    valset = AugSupervisedDataset(
        val_df,
        transform=transforms_dic["val"],
        label_map = dic_classes
        )
    
    testset = AugSupervisedDataset(
        test_df,
        transform=transforms_dic["test"],
        label_map = dic_classes
        )
    
    testset_projection = AugSupervisedDataset(
        test_df,
        transform=transforms_dic["test_projection"],
        label_map = dic_classes
        )

    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection 
    
    
def get_brains(
        dataset_path:str,
        metadata_path: str,
        img_shape: tuple,
        channels: int,
        dic_classes = {'CN':0,'MCI':1,'AD':2},
        n_fold = 5,
        current_fold = 1,
        test_split = 0.2,
        seed = 42):
    
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
    
    return create_datasets(
        dataset_path = dataset_path,
        metadata_path = metadata_path, 
        transforms_dic = transforms_dic,
        dic_classes = dic_classes,
        n_fold = n_fold,
        current_fold = current_fold,
        test_split = test_split,
        seed = seed)


def get_data(args: argparse.Namespace): 

    """ Load dataset based on the parsed arguments """
    
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
        seed = args.seed
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

