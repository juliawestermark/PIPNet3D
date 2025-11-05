"""
Created on Tue Jan 16 16:18:10 2024

@author: lisadesanti
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from copy import deepcopy

from utils import get_args
from make_dataset import get_dataloaders
from model_builder import load_trained_mmpipnet
from test_model import eval_mmpipnet

from test_model import get_local_explanations
from test_model import get_thresholds, eval_ood
from test_model import eval_local_explanations
from test_model import check_empty_prototypes
from vis_pipnet import visualize_topk

import logging
from logger_setup import setup_logger


#%% Global Variables

backbone_dic = {1:"resnet3D_18_kin400", 2:"convnext3D_tiny"}

current_fold = 1
net = backbone_dic[1]
task_performed = "test_mmpipnet"

args = get_args(current_fold, net, task_performed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

setup_logger(level=logging.INFO, log_dir=args.log_dir + "/logs")
logger = logging.getLogger(__name__)

#%% Get Dataloaders for the current_fold

dataloaders = get_dataloaders(args)

trainloader = dataloaders[0]
trainloader_pretraining = dataloaders[1]
trainloader_normal = dataloaders[2] 
trainloader_normal_augment = dataloaders[3]
projectloader = dataloaders[4]
valloader = dataloaders[5]
testloader = dataloaders[6] 
test_projectloader = dataloaders[7]
    
    
#%% Evaluate 3D-PIPNet trained for the current_fold

logger.info("------")
logger.info("PIPNet performances @fold: %s", current_fold)
    
pipnet = load_trained_mmpipnet(args)
pipnet.eval()

# Get the latent space dimensions (needed for prototypes' visualization)
with torch.no_grad():
    # xs1, _ = next(iter(testloader))
    # xs1 = xs1.to(device)
    # print("Input shape: ", xs1.shape, flush=True)
    # proto_features, _, _ = pipnet(xs1)
    modalities_batch = next(iter(testloader))
    xs_list = [xs.to(device) for xs in modalities_batch[:-1]]
    logger.info("Input shape: %s", xs_list[0].shape)
    proto_features_list, _, _, _ = pipnet(xs_list)
    proto_features = proto_features_list[0]
    wshape = proto_features.shape[-1]
    hshape = proto_features.shape[-2]
    dshape = proto_features.shape[-3]
    args.wshape = wshape # needed for calculating image patch size
    args.hshape = hshape # needed for calculating image patch size
    args.dshape = dshape # needed for calculating image patch size
    logger.info("Output shape: %s", proto_features.shape)


#%% Get the Global Explanation
top1, img_prototype_top1, proto_coord_top1 = visualize_topk(
    pipnet, 
    projectloader, 
    args.num_classes, 
    device, 
    'clinical_feedback_global_explanations', 
    args,
    plot=True,
    save=False,
    k=1)

topks, img_prototype, proto_coord = visualize_topk(
    pipnet, 
    projectloader, 
    args.num_classes, 
    device, 
    'visualised_prototypes_topk', 
    args,
    plot=False,
    save=False,
    )

# set weights of prototypes that are never really found in projection set to 0
set_to_zero = []

if topks:
    for prot in topks.keys():
        found = False
        for (i_id, score, mod) in topks[prot]:
            if score > 0.1:
                found = True
        if not found:
            torch.nn.init.zeros_(pipnet.module._classification.weight[:,prot])
            set_to_zero.append(prot)
    logger.info("Weights of prototypes %s are set to zero because it is never detected with similarity>0.1 in the training set", set_to_zero)

logger.info("Classifier weights: %s", pipnet.module._classification.weight)
logger.info("Classifier weights nonzero: %s %s", pipnet.module._classification.weight[pipnet.module._classification.weight.nonzero(as_tuple=True)], (pipnet.module._classification.weight[pipnet.module._classification.weight.nonzero(as_tuple=True)]).shape)
logger.info("Classifier bias: %s", pipnet.module._classification.bias)

# for p in topks.keys(): 
#     print(pipnet.module._classification.weight[:,p])

# Print weights and relevant prototypes per class
for c in range(pipnet.module._classification.weight.shape[0]):
    relevant_ps = []
    proto_weights = pipnet.module._classification.weight[c,:]
    
    for p in range(pipnet.module._classification.weight.shape[1]):
        if proto_weights[p]> 1e-3:
            relevant_ps.append((p, proto_weights[p].item()))

    logger.info("Class %s (%s) has %s relevant prototypes: %s", c, 
          list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)], len(relevant_ps), relevant_ps)


#%% Evaluate PIPNet: 
#    - Classification performances, 
#    - Explanations' size
info = eval_mmpipnet(
    pipnet, 
    testloader, 
    "notused", 
    device)
# for elem in info.items():
#     logger.info(elem)
    
# TODO: the code takes a looooong time to run here.
local_explanations_test, y_preds_test, y_trues_test = get_local_explanations(pipnet, testloader, device, args, plot=True)


#%% Evaluate the prototypes extracted

columns=["detection_rate", "mean_pcc_d", "mean_pcc_h", "mean_pcc_w", "std_pcc_d", "std_pcc_h", "std_pcc_w", "LC"]
logger.info("\nEvaluation of prototypes on test set:")
ps_test_evaluation = eval_local_explanations(pipnet, local_explanations_test, device, args)

ps_test_detections = ps_test_evaluation[0]
ps_test_mean_coords = pd.DataFrame(ps_test_evaluation[1]).transpose().round(decimals=2)
ps_test_std_coords = pd.DataFrame(ps_test_evaluation[2]).transpose().round(decimals=2)
ps_test_lc = pd.Series(ps_test_evaluation[3])
avg_ps_consistency = np.nanmean(np.array([h for h in ps_test_evaluation[3].values()]))
eval_proto_test = pd.concat([ps_test_detections, ps_test_mean_coords, ps_test_std_coords, ps_test_lc], axis=1)
eval_proto_test.columns = columns  

empty_ps = check_empty_prototypes(args, pipnet, img_prototype_top1, proto_coord_top1)


#%% Evaluate OOD Detection
for percent in [95.]:
    logger.info("\nOOD Evaluation for epoch %s with procent of %s", "not used", percent)
    _, _, _, class_thresholds = get_thresholds(
        pipnet, testloader, args.epochs, device, percent)
    
    logger.info("Thresholds: %s", class_thresholds)
    
    # Evaluate with in-distribution data
    id_fraction = eval_ood(
        pipnet, testloader, args.epochs, device, class_thresholds)
    logger.info("ID class threshold ID fraction (TPR) with percent %s: %s", percent, 
          id_fraction)
    
    # Evaluate with out-of-distribution data
    ood_args = deepcopy(args)
    _, _, _, _, _, ood_testloader, _, _ = get_dataloaders(ood_args)
    
    id_fraction = eval_ood(
        pipnet, ood_testloader, args.epochs, device, class_thresholds)
    logger.info("class threshold ID fraction (FPR) with percent %s: %s", percent, 
          id_fraction)                

# Again print final evaluation info
for elem in info.items():
    logger.info(elem)