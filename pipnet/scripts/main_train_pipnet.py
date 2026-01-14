#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:43:20 2023

@author: lisadesanti
Updated for Multimodal PIPNet (MRI + PET)
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from datetime import datetime

# Egna moduler
from utils import set_device, get_optimizer_nn, init_weights_xavier, get_args, Log
from plot_utils import plot_3d_slices
from make_dataset import get_dataloaders
from pipnet import get_network, PIPNet
from train_model import train_pipnet
from test_model import eval_pipnet
from vis_pipnet import visualize_topk

#%% Global Variables

backbone_dic = {1:"resnet3D_18_kin400", 2:"convnext3D_tiny"}

current_fold = 1
net_type = backbone_dic[1]
task_performed = "train_pipnet"

args = get_args(current_fold, net_type, task_performed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Get Dataloaders
print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Dataloaders returnerar nu (inputs_dict, masks_dict, labels)
dataloaders = get_dataloaders(args)
trainloader = dataloaders[0]
trainloader_pretraining = dataloaders[1]
trainloader_normal = dataloaders[2] 
trainloader_normal_augment = dataloaders[3]
projectloader = dataloaders[4]
valloader = dataloaders[5]
testloader = dataloaders[6] 
test_projectloader = dataloaders[7]

log = Log(args.log_dir)

# Check sample (Unpack dictionary)
sample_inputs, sample_masks, sample_labels = next(iter(projectloader))
print(f"Sample loaded. Keys: {sample_inputs.keys()}", flush=True)

#%% Setup Global Masks (Anatomical)

if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)

# 2. Bygg hela sökvägen till mappen
# Struktur: args.model_path / binary / amy_mri / resnet3D_18_kin400
model_save_dir = os.path.join(args.model_path, 'binary', args.model_name, net_type)

# 3. Skapa mappen om den inte redan finns (VIKTIGT!)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"[INFO] Created model directory: {model_save_dir}")

model_save_path = os.path.join(model_save_dir, 'best_pipnet_fold%s'%str(current_fold))

device, device_ids = set_device(args)

# Masks
global_masks = {}

# Loopa över nyckel (t.ex. 'mri', 'amy') och sökväg
for modality, path in args.global_mask_paths.items():
    
    # Default till None om laddning misslyckas eller path saknas
    global_masks[modality] = None
    
    if path and os.path.exists(path):
        print(f"Loading global {modality.upper()} mask from {path}...", flush=True)
        try:
            # 1. Ladda numpy array
            mask_arr = np.load(path).astype(np.float32)
            
            # 2. Konvertera till Tensor och flytta till GPU
            mask_tensor = torch.from_numpy(mask_arr).float().to(device)
            
            # 3. Se till att shape är (1, D, H, W) för broadcasting
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            
            global_masks[modality] = mask_tensor
            
        except Exception as e:
            print(f"ERROR loading {modality} mask from {path}: {e}", flush=True)
    else:
        # Om path är None eller filen inte finns
        if path:
            print(f"WARNING: Path provided for {modality} but file not found: {path}", flush=True)
        else:
            print(f"INFO: No mask path provided for {modality}. Training without mask.", flush=True)
# else:
#     print("WARNING: args.global_mask_paths missing or invalid. No masks loaded.", flush=True)

print(f"Active global masks: {list(global_masks.keys())}", flush=True)

#%% Initiera Multimodal Modell

# get_network returnerar nu dictionaries för backbones och add-ons
(backbones, add_ons, pool_layer, classification_layer, num_prototypes) = get_network(args.out_shape, args)

net = PIPNet(
    num_classes = args.out_shape,
    backbones = backbones,          # Dict
    add_on_layers = add_ons,        # Dict
    pool_layer = pool_layer,
    classification_layer = classification_layer
    )
    
net = net.to(device=device)
net = nn.DataParallel(net, device_ids = device_ids)  

optimizer = get_optimizer_nn(net, args)
optimizer_net = optimizer[0]
optimizer_classifier = optimizer[1] 
params_to_freeze = optimizer[2] 
params_to_train = optimizer[3] 
params_backbone = optimizer[4]   

    
# Initialize or load model (State Dicts)
with torch.no_grad():
    
    if args.state_dict_dir_net != '':
        # Load checkpoint logic (kan behöva anpassas om nyckel-namnen ändrats i state_dict)
        checkpoint = torch.load(args.state_dict_dir_net, map_location = device)
        net.load_state_dict(checkpoint['model_state_dict'], strict = True) 
        print("Pretrained network loaded", flush = True)
        net.module._multiplier.requires_grad = False
        
        try:
            optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
        except:
            pass
        
        # Check classification layer initialization
        if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(net.module._classification.weight).item() < 3.0 and torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(num_prototypes*args.num_classes):
             print("Re-initializing classification layer...", flush = True)
             torch.nn.init.normal_(net.module._classification.weight, mean = 1.0, std = 0.1) 
             torch.nn.init.constant_(net.module._multiplier, val = 2.)
             if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val = 0.)
        else:
            if 'optimizer_classifier_state_dict' in checkpoint.keys():
                optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
        
    else:
        # Initiera add-ons för alla modaliteter
        for mod in net.module._add_ons.keys():
            net.module._add_ons[mod].apply(init_weights_xavier)
            
        torch.nn.init.normal_(net.module._classification.weight, mean = 1.0, std = 0.1) 
        if args.bias:
            torch.nn.init.constant_(net.module._classification.bias, val = 0.)
            
        torch.nn.init.constant_(net.module._multiplier, val = 2.)
        net.module._multiplier.requires_grad = False

        print("Classification layer initialized", flush = True)

# Loss & Scheduler
criterion = nn.NLLLoss(reduction='mean').to(device)

scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_net, 
    T_max = len(trainloader_pretraining)*args.epochs_pretrain, 
    eta_min = args.lr_block/100., 
    last_epoch=-1)

# --- Output Shape Check ---
# Vi kör en batch genom nätet för att se dimensionerna
with torch.no_grad():
    # Packa upp dict och flytta till device
    xs1, xs2, ms1, _ = next(iter(trainloader))
    for k in xs1: xs1[k] = xs1[k].to(device)
    for k in ms1: ms1[k] = ms1[k].to(device)

    proto_out, _, _ = net(xs1, masks=ms1)
    
    # Kolla shape på första modaliteten (t.ex. mri) för loggning
    # proto_out är en dict {'mri': tensor, 'pet': tensor}
    first_feat = list(proto_out.values())[0]
    wshape = first_feat.shape[-1]
    hshape = first_feat.shape[-2]
    dshape = first_feat.shape[-3]
    args.wshape = wshape 
    args.hshape = hshape 
    args.dshape = dshape
    print(f"Output shape (from one modality): {first_feat.shape}", flush=True)

# Logging Setup
if args.out_shape == 2:
    log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_f1', 'almost_sim_nonzeros', 'local_size_all_classes', 'almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
    print("Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance.", flush = True)
else:
    log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_top3_acc', 'almost_sim_nonzeros', 'local_size_all_classes', 'almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')

lrs_pretrain_net = []


print("Training start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#%% 3D-PIPNet Training

#%% PHASE (1): Pretraining Prototypes
for epoch in range(1, args.epochs_pretrain+1):
    
    # --- UPPDATERAD FREEZING LOGIC FÖR MULTIMODAL ---
    # Vi måste loopa över dictionaries nu
    
    for param in params_to_train: param.requires_grad = True
    
    # Unfreeze all add-ons
    for mod in net.module._add_ons.keys():
        for param in net.module._add_ons[mod].parameters():
            param.requires_grad = True
            
    # Freeze classifier
    for param in net.module._classification.parameters():
        param.requires_grad = False
        
    # Backbone freezing logic
    for param in params_to_freeze: param.requires_grad = True 
    for param in params_backbone: param.requires_grad = False 
    
    print("\nPretrain Epoch", epoch, flush = True)
    
    train_info = train_pipnet(
        net, 
        trainloader_pretraining, 
        optimizer_net, 
        optimizer_classifier, 
        scheduler_net, 
        None, 
        criterion, 
        epoch, 
        args.epochs_pretrain, 
        device, 
        pretrain = True, 
        finetune = False,
        mask = global_masks
    )
    
    lrs_pretrain_net += train_info['lrs_net']
    plt.clf()
    plt.plot(lrs_pretrain_net)
    plt.savefig(os.path.join(args.log_dir,'lr_pretrain_net.png'))
    log.log_values('log_epoch_overview', epoch, "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", train_info['loss'])

if args.state_dict_dir_net == '':
    net.eval()
    torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_pretrained'))
    net.train()

#%% PHASE (2): Training PIPNet (Full Training)

# Re-initialize optimizers
optimizer = get_optimizer_nn(net, args)
optimizer_net = optimizer[0]
optimizer_classifier = optimizer[1] 
# Note: params listorna måste vara uppdaterade för dicts (antas funka via rekursion i get_optimizer_nn)
        
scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max = len(trainloader)*args.epochs, eta_min = args.lr_net/100.)

if args.epochs <= 30:
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0 = 5, eta_min = 0.001, T_mult = 1)
else:
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0 = 10, eta_min = 0.001, T_mult = 1)
        
# Grundinställning: Frys allt utom classifier
for param in net.module.parameters(): param.requires_grad = False
for param in net.module._classification.parameters(): param.requires_grad = True

frozen = True
lrs_net = []
lrs_classifier = []
ba_val_old = 0
   
for epoch in range(1, args.epochs + 1): 
                 
    epochs_to_finetune = 3 
    
    # --- LOGIK FÖR FINETUNING / UNFREEZING ---
    if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ''):
        finetune = True
        # Frys allt utom classifier
        for mod in net.module._add_ons.keys():
            for param in net.module._add_ons[mod].parameters(): param.requires_grad = False
            for param in net.module._backbones[mod].parameters(): param.requires_grad = False # Explicit safety
        
    else: 
        finetune = False          
        if frozen:
            # UNFREEZE BACKBONE (Successively)
            if epoch > (args.freeze_epochs):
                for mod in net.module._backbones.keys():
                    for param in net.module._add_ons[mod].parameters(): param.requires_grad = True
                    # Här antar vi att params_to_freeze / params_backbone pekar på rätt parametrar
                    # Om inte, loopa explicit:
                    for param in net.module._backbones[mod].parameters(): param.requires_grad = True
                
                # Generella listor från optimizer-helpern
                for param in params_to_freeze: param.requires_grad = True
                for param in params_to_train: param.requires_grad = True
                for param in params_backbone: param.requires_grad = True   
                frozen = False
            
            # FREEZE FIRST LAYERS ONLY
            else:
                for param in params_to_freeze: param.requires_grad = True 
                for mod in net.module._add_ons.keys():
                    for param in net.module._add_ons[mod].parameters(): param.requires_grad = True
                for param in params_to_train: param.requires_grad = True
                for param in params_backbone: param.requires_grad = False
    
    print("\n Epoch", epoch, "frozen:", frozen, flush = True)  
      
    # Pruning av små vikter i klassificeraren
    if (epoch == args.epochs or epoch%30 == 0) and args.epochs > 1:
        with torch.no_grad():
            torch.set_printoptions(profile = "full")
            net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 0.001, min=0.)) 
            torch.set_printoptions(profile = "default")
    
    train_info = train_pipnet(
        net, 
        trainloader, 
        optimizer_net, 
        optimizer_classifier, 
        scheduler_net, 
        scheduler_classifier, 
        criterion, 
        epoch, 
        args.epochs, 
        device, 
        pretrain = False, 
        finetune = finetune,
        mask = global_masks
    )
    
    lrs_net += train_info['lrs_net']
    lrs_classifier += train_info['lrs_class']
    
    # Evaluate
    eval_info = eval_pipnet(net, valloader, epoch, device, log)
    
    log.log_values('log_epoch_overview',  epoch, eval_info['top1_accuracy'], eval_info['top3_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])
    
    # Save Checkpoints
    ba_val = eval_info["balanced_accuracy"]
    with torch.no_grad():
        net.eval()
        save_dict = {
            'model_state_dict': net.state_dict(), 
            'optimizer_net_state_dict': optimizer_net.state_dict(), 
            'optimizer_classifier_state_dict': optimizer_classifier.state_dict()
        }
        
        torch.save(save_dict, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained'))
        
        if ba_val >= ba_val_old: 
            print(f"New best Balanced Acc: {ba_val:.4f} (prev: {ba_val_old:.4f})", flush=True)
            ba_val_old = ba_val
            torch.save(save_dict, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'best_pipnet_fold%s'%str(current_fold)))
            torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, model_save_path)

        if epoch%30 == 0:
            torch.save(save_dict, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_%s'%str(epoch)))            
    
        # Plot LRs
        plt.clf()
        plt.plot(lrs_net)
        plt.savefig(os.path.join(args.log_dir,'lr_net.png'))
        plt.clf()
        plt.plot(lrs_classifier)
        plt.savefig(os.path.join(args.log_dir,'lr_class.png'))

# Final Save
net.eval()
torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'),'net_trained_last'))

print("Training complete time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Visualization & Pruning of unused prototypes
print("Visualizing Prototypes...", flush=True)
topks, img_prototype, proto_coord = visualize_topk(net, projectloader, args.num_classes, device, 'visualised_prototypes_topk', args, save=False)

# ... (Pruning logic same as before) ...

# set weights of prototypes that are never really found in projection set to 0
set_to_zero = []

if topks:
    for prot in topks.keys():
        found = False
        for (i_id, score) in topks[prot]:
            if score > 0.1:
                found = True
        if not found:
            torch.nn.init.zeros_(net.module._classification.weight[:,prot])
            set_to_zero.append(prot)
    print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)   
    eval_info = eval_pipnet(net, testloader, "notused" + str(args.epochs), device, log)
    log.log_values('log_epoch_overview', "notused"+str(args.epochs), eval_info['top1_accuracy'], eval_info['top3_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], "n.a.", "n.a.")

print("Classifier weights: ", net.module._classification.weight, flush = True)
print("Classifier weights nonzero: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
print("Classifier bias: ", net.module._classification.bias, flush=True)

# Print weights and relevant prototypes per class
for c in range(net.module._classification.weight.shape[0]):
    relevant_ps = []
    proto_weights = net.module._classification.weight[c,:]
    
    for p in range(net.module._classification.weight.shape[1]):
        if proto_weights[p]> 1e-3:
            relevant_ps.append((p, proto_weights[p].item()))
    if args.test_split == 0.:
        print("Class", c, "(", list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)], "):", "has", len(relevant_ps), "relevant prototypes: ", relevant_ps, flush=True)

print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))