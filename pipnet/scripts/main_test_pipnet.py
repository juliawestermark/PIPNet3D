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
from plot_utils import plot_proto_distribution_dynamic
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

# -----------------------------------------------------------------------------
# SNABB-TEST LÄGE: Klipp datasetet "In-Place" (Ingen Subset wrapper!)
# -----------------------------------------------------------------------------
DEBUG_SIZE = None  # Sätt till None eller 0 för att köra allt

if DEBUG_SIZE:
    print(f"\n[DEBUG MODE ACTIVATED] Reducing datasets to {DEBUG_SIZE} samples (In-Place Slice).\n")
    
    def slice_dataset_inplace(loader, num_samples):
        """
        Universell Slicer som anpassar sig efter om du kör Single- eller Multimodal.
        Den garanterar att du får 'valid' data (inte NaN/tomma sökvägar) för de
        modaliteter som är aktiva.
        """
        dataset = loader.dataset
        
        # 1. Ta reda på total längd
        if hasattr(dataset, 'X_paths'):
            # Om det är en dict, ta längden på första listan
            if isinstance(dataset.X_paths, dict):
                first_key = list(dataset.X_paths.keys())[0]
                full_len = len(dataset.X_paths[first_key])
            else:
                full_len = len(dataset.X_paths)
        else:
            full_len = len(dataset)
            
        limit = min(full_len, num_samples)
        
        print(f"  -> Scanning {type(dataset).__name__} ({full_len} samples) to find valid data...")

        # ---------------------------------------------------------
        # 2. Identifiera "Bra" Index (som har data för dina modaliteter)
        # ---------------------------------------------------------
        valid_indices_priority = [] # T.ex. de som har PET (om vi kör multimodalt)
        valid_indices_standard = [] # T.ex. de som bara har MRI
        
        # Ta reda på vilka modaliteter som finns i datasetet JUST NU
        paths_obj = dataset.X_paths if hasattr(dataset, 'X_paths') else []
        
        # Kolla om vi hanterar en dict (Multimodal) eller lista (Single)
        is_dict = isinstance(paths_obj, dict)
        
        # Identifiera "Sällsynta" nycklar vi vill prioritera (för att inte missa dem i debug)
        priority_keys = ['amy'] 
        has_priority_key = False
        
        if is_dict:
            keys = list(paths_obj.keys()) # T.ex. ['mri'] eller ['mri', 'amy']
            # Kolla om någon av nycklarna är "priority"
            for k in keys:
                if any(pk in k.lower() for pk in priority_keys):
                    has_priority_key = True
        else:
            keys = None # Single modal (bara en lista)
            
        # --- SCANNA DATASETET ---
        for idx in range(full_len):
            is_valid_sample = False
            is_priority_sample = False
            
            if is_dict:
                # MULTIMODAL / DICT
                # Kolla att åtminstone en modalitet har en path (eller alla)
                # Här kör vi strategin: Om en path finns är det ett valid sample
                row_has_data = False
                row_has_priority = False
                
                for k in keys:
                    p = paths_obj[k][idx]
                    # Kolla om path är "riktig" (sträng, inte nan, inte tom)
                    if isinstance(p, str) and len(p) > 2 and "nan" not in p.lower():
                        row_has_data = True
                        if any(pk in k.lower() for pk in priority_keys):
                            row_has_priority = True
                
                if row_has_data:
                    if row_has_priority:
                        valid_indices_priority.append(idx)
                    else:
                        valid_indices_standard.append(idx)

            else:
                # SINGLE MODAL / LISTA
                p = paths_obj[idx] if hasattr(paths_obj, 'iloc') else paths_obj[idx]
                if isinstance(p, str) and len(p) > 2 and "nan" not in p.lower():
                    valid_indices_standard.append(idx)

        # ---------------------------------------------------------
        # 3. Välj ut indexen (Balansera)
        # ---------------------------------------------------------
        selected_indices = []
        
        # Om vi hittade prioriterad data (t.ex. PET), fyll upp halva kvoten med den
        if valid_indices_priority:
            take_n = min(len(valid_indices_priority), limit // 2 if valid_indices_standard else limit)
            selected_indices.extend(valid_indices_priority[:take_n])
            print(f"     Prioritized {take_n} samples containing AMY.")
            
        # Fyll på resten med standard (eller mer priority om det finns)
        remaining = limit - len(selected_indices)
        
        # Lägg till standard (t.ex. MRI)
        if remaining > 0 and valid_indices_standard:
            take_std = min(len(valid_indices_standard), remaining)
            selected_indices.extend(valid_indices_standard[:take_std])
            remaining -= take_std
            
        # Om fortfarande plats och vi har fler priority över
        if remaining > 0 and len(valid_indices_priority) > len(selected_indices):
            # Hitta de vi inte tog
            used_set = set(selected_indices)
            rest_prio = [i for i in valid_indices_priority if i not in used_set]
            selected_indices.extend(rest_prio[:remaining])

        # Sortera index för ordningens skull
        selected_indices.sort()
        actual_limit = len(selected_indices)
        
        print(f"  -> Selected {actual_limit} valid samples based on available modalities.")

        # ---------------------------------------------------------
        # 4. Skapa Subset & Patcha (Detta känner du igen)
        # ---------------------------------------------------------
        subset = torch.utils.data.Subset(dataset, selected_indices)
        
        def index_slice(obj, indices):
            if isinstance(obj, dict):
                return {k: index_slice(v, indices) for k, v in obj.items()}
            if hasattr(obj, 'iloc'): # Pandas
                return obj.iloc[indices]
            if isinstance(obj, list): # Lista
                return [obj[i] for i in indices]
            if hasattr(obj, 'numpy'): # Tensor
                return obj[indices]
            return obj[:len(indices)]

        # Kopiera över X_paths (viktigast!)
        if hasattr(dataset, 'X_paths'):
            subset.X_paths = index_slice(dataset.X_paths, selected_indices)
            
        # Kopiera ys
        if hasattr(dataset, 'ys'):
            subset.ys = index_slice(dataset.ys, selected_indices)
            
        # Kopiera metadata rakt av
        if hasattr(dataset, 'class_to_idx'):
            subset.class_to_idx = dataset.class_to_idx
        if hasattr(dataset, 'col_names'):
            subset.col_names = dataset.col_names
        if hasattr(dataset, 'image_paths'):
             subset.image_paths = index_slice(dataset.image_paths, selected_indices)

        # Skapa DataLoader
        new_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False, 
            num_workers=4, # 0 workers är säkrast vid debug
            pin_memory=loader.pin_memory
        )
        
        return new_loader

    # Applicera på dina loaders
    # Eftersom vi ändrar dataset-objektet, slår detta igenom överallt
    projectloader = slice_dataset_inplace(projectloader, DEBUG_SIZE)
    testloader = slice_dataset_inplace(testloader, DEBUG_SIZE)
    
    # Om du använder OOD loaders senare, kör funktionen på dem också!
# -----------------------------------------------------------------------------
    
    
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

# print("--- PRUNING: Behåller bara de X viktigaste prototyperna per klass ---")
# # Hur många vill du ha max per klass? T.ex. 5 st ger extrem tydlighet.
# TOP_K_PER_CLASS = 5 

# with torch.no_grad():
#     weights = pipnet.module._classification.weight
#     # Skapa en mask med bara nollor
#     new_mask = torch.zeros_like(weights)
    
#     for c in range(weights.shape[0]):
#         # Hämta vikterna för klass c
#         class_weights = weights[c, :]
        
#         # Hitta index för de absolut största vikterna
#         # topk returnerar (values, indices)
#         _, top_indices = torch.topk(class_weights, TOP_K_PER_CLASS)
        
#         # Sätt en 1:a i masken för dessa index
#         new_mask[c, top_indices] = 1.0

#     # Multiplicera vikterna med masken -> Allt annat blir 0.0
#     pipnet.module._classification.weight *= new_mask

# print(f"Klassificeringslagret är nu rensat. Endast topp {TOP_K_PER_CLASS} prototyper per klass används.")

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
    print("\n--- ROBUST SPATIAL PRUNING: Top-K (Kräver 0 träffar för radering) ---", flush=True)

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
            
            # --- ÄNDRING 1: Endast noll träffar raderas! ---
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

    print(f"\n[INFO] Rensade {len(spatial_zeros)} prototyper (som hade 0 träffar i hjärnan på {K_TO_CHECK} försök).")

    if len(pruned_explanations) > 0:
        print("[INFO] Skapar samlingsbild för prunade prototyper...")
        
        # --- ÄNDRING 2: Bygg en garanterat komplett referensbild ---
        print("[INFO] Letar upp giltiga bakgrundsbilder för plottning...")
        xs_ref_perfect = {}
        mods_needed = list(modality_indices.keys())
        
        for xs_batch, ms_batch, _ in projectloader:
            for mod in mods_needed:
                if mod not in xs_ref_perfect:
                    if ms_batch[mod][0].item() == 1.0:
                        xs_ref_perfect[mod] = xs_batch[mod].clone()
            
            # Avbryt loopen så fort vi har samlat ihop minst en bra bild per modalitet!
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
        print(f"[INFO] Samlingsbild sparad till: {pruned_save_path}")

    print("---------------------------------------------------------------", flush=True)

# set weights of prototypes that are never really found in projection set to 0
set_to_zero = []

if topks:
    for prot in topks.keys():
        
        # --- NYTT: Hitta vilken modalitet prototypen tillhör för att få rätt tröskel ---
        current_thr = 0.1 # Fallback
        
        # Om args.threshold är en dictionary (t.ex. {'mri': 0.1, 'amy': 0.01})
        if isinstance(args.threshold, dict):
            for mod, (start, end) in modality_indices.items():
                if start <= prot < end:
                    current_thr = args.threshold[mod]
                    break
        # Om det bara är en float (t.ex. 0.1)
        elif args.threshold is not None:
            current_thr = float(args.threshold)
        # -----------------------------------------------------------------------------

        found = False
        for (i_id, score) in topks[prot]:
            # HÄR ÄR ÄNDRINGEN: Använd current_thr istället för hårdkodat 0.1
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

# Kör evalueringen
ps_test_evaluation = eval_local_explanations(pipnet, local_explanations_test, device, args)

# Bygg DataFrame som du gjorde innan
ps_test_detections = ps_test_evaluation[0]
ps_test_mean_coords = pd.DataFrame(ps_test_evaluation[1]).transpose().round(decimals=2)
ps_test_std_coords = pd.DataFrame(ps_test_evaluation[2]).transpose().round(decimals=2)
ps_test_lc = pd.Series(ps_test_evaluation[3])

eval_proto_test = pd.concat([ps_test_detections, ps_test_mean_coords, ps_test_std_coords, ps_test_lc], axis=1)
eval_proto_test.columns = columns  

# --- HÄR ÄR FIXEN ---
# Filtrera fram endast de prototyper som faktiskt detekterades minst en gång
active_protos = eval_proto_test[eval_proto_test["detection_rate"] > 0]

# Beräkna snittet bara på dessa
avg_ps_consistency = active_protos["LC"].mean()

# Spara till fil (Hela tabellen är bra att spara för att se vilka som är 0)
csv_path = os.path.join(args.log_dir, f"prototype_metrics_fold{current_fold}.csv")
eval_proto_test.to_csv(csv_path)
print(f"\n[INFO] Prototype metrics saved to: {csv_path}", flush=True)

# Skriv ut sammanfattning
print("\n--- Prototype Evaluation Summary ---", flush=True)
print(f"Total prototypes: {len(eval_proto_test)}")
print(f"Active prototypes (detected > 0 times): {len(active_protos)}")
print(f"Average Local Consistency (Active only): {avg_ps_consistency:.4f}", flush=True)

# Om du vill se Active Protos i terminalen istället för bara de första 10 (som kan vara nollor)
print("\nTop 10 Active Prototypes by Detection Rate:")
print(active_protos.sort_values(by="detection_rate", ascending=False).head(10))
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
