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

import torch.nn.functional as F
from tqdm import tqdm

from utils import get_args
from make_dataset import get_dataloaders
from model_builder import load_trained_pipnet
from test_model import eval_pipnet

from test_model import get_local_explanations
from test_model import get_thresholds, eval_ood
from test_model import eval_local_explanations
from test_model import check_empty_prototypes
from vis_pipnet import visualize_topk
from vis_pipnet import plot_local_explanation



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

print(f"Threshold: {args.threshold}")

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
    proto_features_dict, _, _ = pipnet(xs1, masks=ms1)
    
    # --- UPDATE: Get dimensions from first available modality ---
    first_mod = list(proto_features_dict.keys())[0]
    proto_features = proto_features_dict[first_mod]
    
    wshape = proto_features.shape[-1]
    hshape = proto_features.shape[-2]
    dshape = proto_features.shape[-3]
    args.wshape = wshape 
    args.hshape = hshape 
    args.dshape = dshape 
    print(f"Output shape ({first_mod}): {proto_features.shape}", flush=True)

# 2. Calculate offsets dynamically
modalities = pipnet.module.modalities # e.g. ['mri', 'amy']
modality_indices = {} # {'mri': (0, 512), 'amy': (512, 1024), ...}
current_offset = 0

for mod in modalities:
    add_on_module = pipnet.module._add_ons[mod]
    num_protos = 0
    # Locate Conv3d layer to get exact channel count
    for m in add_on_module.modules():
        if isinstance(m, torch.nn.Conv3d):
            num_protos = m.out_channels
            break
    
    # Fallback logic
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
    k=1,
    threshold=args.threshold)

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
    threshold=args.threshold
    )

if args.prune_k_checks:
    print("\n--- ROBUST SPATIAL PRUNING: Top-K (Requires 0 hits for deletion) ---", flush=True)

    K_TO_CHECK = args.prune_k_checks            
    INTENSITY_THRESHOLD = 0.05 

    latent_shape = (args.dshape, args.hshape, args.wshape)
    spatial_zeros = []
    pruned_explanations = {}

    pipnet.eval()
    with torch.no_grad():
        
        pbar = tqdm(topks.items(), desc="Pruning Protos", mininterval=2.0, ascii=True)
        
        for prot_idx, img_list in pbar:
            mod_key = None
            local_prot_idx = 0
            for mod, (start, end) in modality_indices.items():
                if start <= prot_idx < end:
                    mod_key = mod
                    local_prot_idx = prot_idx - start 
                    break
                    
            if mod_key is None: continue 
                
            brain_hits = 0
            images_checked = 0
            best_d, best_h, best_w = 0, 0, 0
            
            for (img_idx, score) in img_list[:K_TO_CHECK]:
                xs, _, _ = projectloader.dataset[img_idx]
                input_dict = {k: v.unsqueeze(0).to(device) for k, v in xs.items()}
                img_tensor = input_dict[mod_key] 
                
                threshold_val = img_tensor.max() * INTENSITY_THRESHOLD
                brain_mask = (img_tensor > threshold_val).float()
                pooled_mask = F.adaptive_max_pool3d(brain_mask, output_size=latent_shape)
                pooled_mask_bool = pooled_mask[0, 0] > 0 
                
                proto_features_dict, _, _ = pipnet(input_dict)
                feature_map = proto_features_dict[mod_key][0, local_prot_idx]
                
                flat_idx = torch.argmax(feature_map)
                d, h, w = np.unravel_index(flat_idx.cpu().numpy(), latent_shape)
                
                if pooled_mask_bool[d, h, w]:
                    brain_hits += 1
                    
                if images_checked == 0:
                    best_d, best_h, best_w = d, h, w
                    img_shape = img_tensor.shape[2:] # (D, H, W)
                    
                images_checked += 1

                if brain_hits > 0:
                    break
            
            # --- CHANGE 1: Only prototypes with zero hits are deleted! ---
            if brain_hits == 0 and images_checked > 0:
                pipnet.module._classification.weight[:, prot_idx] = 0.0
                spatial_zeros.append(prot_idx)
                
                stride_d = img_shape[0] // latent_shape[0]
                stride_h = img_shape[1] // latent_shape[1]
                stride_w = img_shape[2] // latent_shape[2]
                
                d_min = best_d * stride_d
                d_max = min((best_d + 1) * stride_d, img_shape[0])
                
                h_min = best_h * stride_h
                h_max = min((best_h + 1) * stride_h, img_shape[1])
                
                w_min = best_w * stride_w
                w_max = min((best_w + 1) * stride_w, img_shape[2])
                
                ps_coord = (d_min, d_max, h_min, h_max, w_min, w_max)
                fake_score = float(brain_hits) 
                
                pruned_explanations[prot_idx] = (ps_coord, fake_score)

            pbar.set_postfix({'Pruned': len(spatial_zeros)})

    print(f"\n[INFO] Pruned {len(spatial_zeros)} prototypes (which had 0 brain hits in {K_TO_CHECK} attempts).")

    if len(pruned_explanations) > 0:
        print("[INFO] Creating summary plot for pruned prototypes...")
        
        # --- CHANGE 2: Ensure a complete reference image is built ---
        print("[INFO] Finding valid background images for plotting...")
        xs_ref_perfect = {}
        mods_needed = list(modality_indices.keys())
        
        for xs_batch, ms_batch, _ in projectloader:
            for mod in mods_needed:
                if mod not in xs_ref_perfect:
                    if ms_batch[mod][0].item() == 1.0:
                        xs_ref_perfect[mod] = xs_batch[mod].clone()
            
            # Stop loop once we have at least one good image per modality
            if len(xs_ref_perfect) == len(mods_needed):
                break
        
        pruned_save_path = os.path.join(args.log_dir, f"pruned_prototypes_fold{current_fold}.png")
        
        plot_local_explanation(
            xs=xs_ref_perfect, 
            local_explanation=pruned_explanations, 
            modality_offsets=modality_indices, 
            title="Pruned Prototypes (100% Outside Brain)", 
            footer=f"K={K_TO_CHECK} (Zero brain hits)", 
            save_path=pruned_save_path
        )
        print(f"[INFO] Summary plot saved to: {pruned_save_path}")

    print("---------------------------------------------------------------", flush=True)

# Set weights of prototypes never detected above threshold to zero
set_to_zero = []

if topks:
    for prot in topks.keys():
        
        # Determine modality of prototype to apply correct threshold
        current_thr = 0.1 # Fallback
        
        # If args.threshold is a dictionary (e.g. {'mri': 0.1, 'amy': 0.01})
        if isinstance(args.threshold, dict):
            for mod, (start, end) in modality_indices.items():
                if start <= prot < end:
                    current_thr = args.threshold[mod]
                    break
        # If it is a float
        elif args.threshold is not None:
            current_thr = float(args.threshold)

        found = False
        for (i_id, score) in topks[prot]:
            # Use current_thr instead of hardcoded 0.1
            if score > current_thr:
                found = True
                
        if not found:
            torch.nn.init.zeros_(pipnet.module._classification.weight[:,prot])
            set_to_zero.append(prot)
            
    print("Weights of prototypes", set_to_zero, "are set to zero because they were never detected above their threshold.", flush=True)

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
    modality_ranges=modality_indices,
    threshold=args.threshold
    )

for elem in info.items():
    print(elem)
    
print("\n--- Getting Local Explanations ---", flush=True)
local_explanations_test, y_preds_test, y_trues_test = get_local_explanations(pipnet, testloader, device, args, plot=True, threshold=args.threshold)


#%% Evaluate the prototypes extracted
print("\n--- Evaluating Extracted Prototypes ---", flush=True)

columns=["detection_rate", "mean_pcc_d", "mean_pcc_h", "mean_pcc_w", "std_pcc_d", "std_pcc_h", "std_pcc_w", "LC"]

# Run evaluation
ps_test_evaluation = eval_local_explanations(pipnet, local_explanations_test, device, args)

ps_test_detections = ps_test_evaluation[0]
ps_test_mean_coords = pd.DataFrame(ps_test_evaluation[1]).transpose().round(decimals=2)
ps_test_std_coords = pd.DataFrame(ps_test_evaluation[2]).transpose().round(decimals=2)
ps_test_lc = pd.Series(ps_test_evaluation[3])

eval_proto_test = pd.concat([ps_test_detections, ps_test_mean_coords, ps_test_std_coords, ps_test_lc], axis=1)
eval_proto_test.columns = columns  

# Filter for prototypes detected at least once
active_protos = eval_proto_test[eval_proto_test["detection_rate"] > 0]

# Calculate mean LC only for active prototypes
avg_ps_consistency = active_protos["LC"].mean()

# Save to file
csv_path = os.path.join(args.log_dir, f"prototype_metrics_fold{current_fold}.csv")
eval_proto_test.to_csv(csv_path)
print(f"\n[INFO] Prototype metrics saved to: {csv_path}", flush=True)

# Print summary
print("\n--- Prototype Evaluation Summary ---", flush=True)
print(f"Total prototypes: {len(eval_proto_test)}")
print(f"Active prototypes (detected > 0 times): {len(active_protos)}")
print(f"Average Local Consistency (Active only): {avg_ps_consistency:.4f}", flush=True)

print("\nTop 10 Active Prototypes by Detection Rate:")
print(active_protos.sort_values(by="detection_rate", ascending=False).head(10))

print("\n--- Multimodal Contribution Analysis (Dynamic) ---", flush=True)

# 1. Get weights and modalities
weights = pipnet.module._classification.weight.detach().cpu()

# 3. Analyze per class
for c in range(weights.shape[0]):
    class_name = list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)]
    
    class_weights = weights[c, :]
    total_importance_sum = 0
    modality_stats = {}

    for mod, (start, end) in modality_indices.items():
        w_mod = class_weights[start:end]
        
        count_used = (w_mod > 1e-3).sum().item()
        importance = w_mod[w_mod > 1e-3].sum().item()
        
        modality_stats[mod] = {'count': count_used, 'importance': importance}
        total_importance_sum += importance

    print(f"\nClass {class_name} Analysis:")
    if total_importance_sum == 0: total_importance_sum = 1e-9 # Avoid division by zero

    for mod in modalities:
        stats = modality_stats[mod]
        imp_percent = (stats['importance'] / total_importance_sum) * 100
        print(f"  {mod.upper():<5}: {stats['count']:>3} protos used | Importance: {stats['importance']:.4f} ({imp_percent:.1f}%)")

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
    ood_args = deepcopy(args)
    ood_dataloaders = get_dataloaders(ood_args)
    ood_testloader = ood_dataloaders[6] 
    
    id_fraction = eval_ood(
        pipnet, ood_testloader, args.epochs, device, class_thresholds)
    print("class threshold ID fraction (FPR) with percent", percent,":", 
          id_fraction, flush=True)

print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))