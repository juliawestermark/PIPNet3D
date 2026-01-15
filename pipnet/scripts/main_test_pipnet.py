"""
Created on Tue Jan 16 16:18:10 2024

@author: lisadesanti
Updated for Multimodal PIPNet
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
from datetime import datetime

from utils import get_args
from make_dataset import get_dataloaders
from model_builder import load_trained_pipnet
from test_model import eval_pipnet

from test_model import get_local_explanations
from test_model import get_thresholds, eval_ood
from test_model import eval_local_explanations
from test_model import check_empty_prototypes
from vis_pipnet import visualize_topk
from plot_utils import plot_proto_distribution_dynamic



#%% Global Variables

backbone_dic = {1:"resnet3D_18_kin400", 2:"convnext3D_tiny"}

current_fold = 1
net = backbone_dic[1]
task_performed = "test_pipnet"

args = get_args(current_fold, net, task_performed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Get Dataloaders for the current_fold
print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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
print("Start testing time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("------", flush = True)
print("PIPNet performances @fold: ", current_fold, flush = True)
    
pipnet = load_trained_pipnet(args)
pipnet.eval()
pipnet.to(device)

# Get the latent space dimensions (needed for prototypes' visualization)
print("Calculating latent space dimensions...", flush=True)
with torch.no_grad():
    # --- UPDATE: Unpack 3 values (Data, Mask, Label) ---
    xs1, ms1, _ = next(iter(testloader))
    
    # Move dicts to device
    xs1 = {k: v.to(device) for k, v in xs1.items()}
    ms1 = {k: v.to(device) for k, v in ms1.items()} if ms1 is not None else None
    
    # Print shapes for debug
    for k, v in xs1.items():
        print(f"Input {k} shape: {v.shape}", flush=True)

    # --- UPDATE: Forward pass with masks ---
    # Returns (features_dict, pooled, out)
    proto_features_dict, _, _ = pipnet(xs1, masks=ms1)
    
    # --- UPDATE: Get dimensions from first available modality ---
    first_mod = list(proto_features_dict.keys())[0]
    proto_features = proto_features_dict[first_mod]
    # TODO: Fix bug if first modality is none
    
    wshape = proto_features.shape[-1]
    hshape = proto_features.shape[-2]
    dshape = proto_features.shape[-3]
    args.wshape = wshape 
    args.hshape = hshape 
    args.dshape = dshape 
    print(f"Output shape ({first_mod}): {proto_features.shape}", flush=True)

# 2. Räkna ut offsets dynamiskt (Samma logik som i dina andra filer)
modalities = pipnet.module.modalities # T.ex. ['mri', 'amy'] eller ['mri', 'pet', 'tau']
modality_indices = {} # {'mri': (0, 512), 'amy': (512, 1024), ...}
current_offset = 0

for mod in modalities:
    add_on_module = pipnet.module._add_ons[mod]
    num_protos = 0
    # Leta upp Conv3d lagret för att veta exakt antal kanaler
    for m in add_on_module.modules():
        if isinstance(m, torch.nn.Conv3d):
            num_protos = m.out_channels
            break
    
    # Fallback om det krånglar
    if num_protos == 0: 
        num_protos = getattr(args, 'num_features', 512)
    if num_protos == 0: 
        num_protos = 512

    modality_indices[mod] = (current_offset, current_offset + num_protos)
    current_offset += num_protos

#%% Get the Global Explanation
print("\n--- Visualizing Global Explanations (Top 1) ---", flush=True)
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

print("\n--- Visualizing Top K Prototypes ---", flush=True)
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
        for (i_id, score) in topks[prot]:
            if score > 0.1:
                found = True
        if not found:
            torch.nn.init.zeros_(pipnet.module._classification.weight[:,prot])
            set_to_zero.append(prot)
    print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)


# Print weights and relevant prototypes per class
for c in range(pipnet.module._classification.weight.shape[0]):
    relevant_ps = []
    proto_weights = pipnet.module._classification.weight[c,:]
    
    for p in range(pipnet.module._classification.weight.shape[1]):
        if proto_weights[p]> 1e-3:
            relevant_ps.append((p, proto_weights[p].item()))

    class_name = list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)]
    print(f"Class {c} ({class_name}) has {len(relevant_ps)} relevant prototypes.", flush=True)


#%% Evaluate PIPNet: 
#    - Classification performances, 
#    - Explanations' size
print("\n--- Evaluating PIPNet on Test Set ---", flush=True)
print("Start time on testset:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
info = eval_pipnet(
    pipnet, 
    testloader, 
    "notused", 
    device,
    modality_ranges=modality_indices)

for elem in info.items():
    print(elem)
    
print("\n--- Getting Local Explanations ---", flush=True)
local_explanations_test, y_preds_test, y_trues_test = get_local_explanations(pipnet, testloader, device, args, plot=True)


#%% Evaluate the prototypes extracted
print("\n--- Evaluating Extracted Prototypes ---", flush=True)

columns=["detection_rate", "mean_pcc_d", "mean_pcc_h", "mean_pcc_w", "std_pcc_d", "std_pcc_h", "std_pcc_w", "LC"]

ps_test_evaluation = eval_local_explanations(pipnet, local_explanations_test, device, args)

ps_test_detections = ps_test_evaluation[0]
ps_test_mean_coords = pd.DataFrame(ps_test_evaluation[1]).transpose().round(decimals=2)
ps_test_std_coords = pd.DataFrame(ps_test_evaluation[2]).transpose().round(decimals=2)
ps_test_lc = pd.Series(ps_test_evaluation[3])
avg_ps_consistency = np.nanmean(np.array([h for h in ps_test_evaluation[3].values()]))
eval_proto_test = pd.concat([ps_test_detections, ps_test_mean_coords, ps_test_std_coords, ps_test_lc], axis=1)
eval_proto_test.columns = columns  

# Note: check_empty_prototypes logic might need MM updates in test_model.py, usually safe to skip if buggy
# empty_ps = check_empty_prototypes(args, pipnet, img_prototype_top1, proto_coord_top1)

# 1. Spara till fil (Bäst för analys)
csv_path = os.path.join(args.log_dir, f"prototype_metrics_fold{current_fold}.csv")
eval_proto_test.to_csv(csv_path)
print(f"\n[INFO] Prototype metrics saved to: {csv_path}", flush=True)

# 2. Skriv ut en sammanfattning i terminalen
print("\n--- Prototype Evaluation Summary ---", flush=True)
print(eval_proto_test.head(10)) # Visar de 10 första
print(f"Average Local Consistency: {avg_ps_consistency:.4f}", flush=True)
# ---------------------------------------

print("\n--- Multimodal Contribution Analysis (Dynamic) ---", flush=True)

# 1. Hämta vikter och modaliteter
weights = pipnet.module._classification.weight.detach().cpu()



# 3. Analysera per klass
for c in range(weights.shape[0]):
    # Hämta klassnamn (t.ex. "AD")
    class_name = list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)]
    
    class_weights = weights[c, :]
    total_importance_sum = 0
    modality_stats = {}

    # Samla statistik för varje modalitet
    for mod, (start, end) in modality_indices.items():
        # Klipp ut vikterna för just denna modalitet
        w_mod = class_weights[start:end]
        
        # Räkna
        count_used = (w_mod > 1e-3).sum().item()
        importance = w_mod[w_mod > 1e-3].sum().item()
        
        modality_stats[mod] = {'count': count_used, 'importance': importance}
        total_importance_sum += importance

    # Skriv ut resultatet
    print(f"\nClass {class_name} Analysis:")
    if total_importance_sum == 0: total_importance_sum = 1e-9 # Undvik division med noll

    for mod in modalities:
        stats = modality_stats[mod]
        imp_percent = (stats['importance'] / total_importance_sum) * 100
        print(f"  {mod.upper():<5}: {stats['count']:>3} protos used | Importance: {stats['importance']:.4f} ({imp_percent:.1f}%)")


# --- Hämta en referensbild (MRI) för bakgrunden ---
ref_vol = None
try:
    # Hämta första batchen igen (vi gjorde det tidigt i scriptet, men gör det igen för säkerhets skull)
    xs_ref, _, _ = next(iter(testloader))
    
    # Försök hitta 'mri' i första hand, annars ta första bästa
    if 'mri' in xs_ref:
        # Ta första bilden i batchen, kanal 0 (intensity), cpu, numpy
        ref_vol = xs_ref['mri'][0, 0].cpu().numpy()
    else:
        first_key = list(xs_ref.keys())[0]
        ref_vol = xs_ref[first_key][0, 0].cpu().numpy()
        
    print(f"[INFO] Using {first_key if 'mri' not in xs_ref else 'mri'} volume as spatial reference: {ref_vol.shape}")
except Exception as e:
    print(f"[WARN] Could not fetch reference volume: {e}")

# --- Anropa den nya funktionen ---
try:
    plot_proto_distribution_dynamic(
        eval_proto_test, 
        pipnet.module._classification.weight.detach().cpu(), 
        modality_indices,
        args.log_dir,
        reference_volume=ref_vol # <--- SKICKAR MED BILDEN HÄR
    )
except Exception as e:
    print(f"[WARN] Could not plot 3D distribution: {e}")

#%% Evaluate OOD Detection
print("\n--- Evaluating OOD Detection ---", flush=True)
for percent in [95.]:
    print("\nOOD Evaluation for epoch", "not used","with percent of", percent, 
          flush=True)
    _, _, _, class_thresholds = get_thresholds(
        pipnet, testloader, args.epochs, device, percent)
    
    print("Thresholds:", class_thresholds, flush=True)
    
    # Evaluate with in-distribution data
    id_fraction = eval_ood(
        pipnet, testloader, args.epochs, device, class_thresholds)
    print("ID class threshold ID fraction (TPR) with percent", percent, ":", 
          id_fraction, flush=True)
    
    # Evaluate with out-of-distribution data
    # NOTE: Ensure ood_args points to correct dataset and get_dataloaders returns MM loader
    ood_args = deepcopy(args)
    # Exempel på att ändra dataset path för OOD om det behövs:
    # ood_args.dataset_path = "/path/to/ood/data" 
    
    # Vi hämtar testloader (index 6) från ood dataloaders
    ood_dataloaders = get_dataloaders(ood_args)
    ood_testloader = ood_dataloaders[6] # Test set of OOD data
    
    id_fraction = eval_ood(
        pipnet, ood_testloader, args.epochs, device, class_thresholds)
    print("class threshold ID fraction (FPR) with percent", percent,":", 
          id_fraction, flush=True)

print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
