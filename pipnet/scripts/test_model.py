import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import monai.transforms as transforms
import re
import textwrap

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from scipy.stats import entropy

from utils import Log
from utils import topk_accuracy
from vis_pipnet import get_patch_size_dynamic, get_img_coordinates, plot_local_explanation
from plot_utils import plot_3d_slices, plot_rgb_slices, generate_rgb_array



@torch.no_grad()
def eval_pipnet(
        net,
        test_loader,
        epoch,
        device,
        log = None,  
        progress_prefix: str = 'Eval Epoch',
        modality_ranges: dict = None,
        threshold = None
        ) -> dict:
    
    net = net.to(device)
    net.eval()
    info = dict()
    
    cm = np.zeros((net.module._num_classes, net.module._num_classes), dtype = int)

    global_top1acc = 0.
    global_top3acc = 0.
    global_sim_anz = 0.
    global_anz = 0.
    local_size_total = 0.
    local_size_total_sum_of_mod = 0.
    y_trues = []
    y_preds = []
    y_preds_classes = []
    y_probs_all = []
    abstained = 0
    
    # --- PREPARATION FOR COMPARABILITY ---
    # Create a mask of which prototypes are "alive" (weight > 1e-3).
    # This prevents counting prototypes that have been zeroed out (sparsity).
    # Shape: [Total_Num_Prototypes] -> [1, 0, 1, 1, ...]
    cls_weights = net.module._classification.weight
    proto_is_globally_active = torch.gt(torch.abs(cls_weights), 1e-3).any(dim=0).float().to(device)
    # -------------------------------------

    modality_stats = {}
    if modality_ranges:
        for mod in modality_ranges.keys():
            modality_stats[mod] = {'active_protos': 0., 'valid_samples': 0.}
    
    test_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc=progress_prefix+' %s'%epoch, mininterval=5., ncols=0)
    
    for i, (xs, ms, ys) in test_iter:
        
        ys = ys.to(device)
        xs = {k: v.to(device) for k, v in xs.items()}
        ms_device = {k: v.to(device) for k, v in ms.items()} if ms is not None else None
        
        with torch.no_grad():
            _, pooled, out = net(xs, masks=ms_device, inference = True, threshold=threshold)
            
            max_out_score, ys_pred = torch.max(out, dim=1)
            
            probs = F.softmax(out, dim=1) 
            y_probs_all.extend(probs.detach().cpu().numpy())

            ys_pred_scores = torch.amax(F.softmax((torch.log1p(out**net.module._classification.normalization_multiplier)), dim = 1), dim = 1) 
            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))  

            repeated_weight = net.module._classification.weight.unsqueeze(1).repeat(1, pooled.shape[0], 1)
            sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(pooled*repeated_weight), 1e-3).float(), dim = 2).float()
            local_size = torch.count_nonzero(torch.gt(torch.relu((pooled*repeated_weight) - 1e-3).sum(dim = 1), 0.).float(), dim = 1).float()
            local_size_total += local_size.sum().item()
            correct_class_sim_scores_anz = torch.diagonal(torch.index_select(sim_scores_anz, dim = 0, index = ys_pred), 0)
            global_sim_anz += correct_class_sim_scores_anz.sum().item()
            almost_nz = torch.count_nonzero(torch.gt(torch.abs(pooled), 1e-3).float(), dim = 1).float()
            global_anz += almost_nz.sum().item()
            
            # 2. Prepare calculation of Local Size (LS)
            # Based entirely on the weights for the class that was actually chosen.
            # This is the "True Explanation".
            weights_for_pred = net.module._classification.weight[ys_pred] # [Batch, Total_Protos]
            all_contributions = pooled * weights_for_pred # [Batch, Total_Protos]
            
            # Mask: Which prototypes contributed more than the threshold?
            is_relevant_global = torch.gt(torch.abs(all_contributions), 1e-3).float()

            # Variable to track Global LS for this batch
            # Reset and built via modalities for maximum control
            batch_ls_global = torch.zeros(pooled.shape[0]).to(device)

            # --- LOOP OVER MODALITIES ---
            if modality_ranges:
                for mod, (start, end) in modality_ranges.items():
                    # A. Missing Data Mask
                    if ms_device is not None and mod in ms_device:
                        # Ensure 1D-vector [Batch]
                        current_mask = ms_device[mod].view(-1)
                    else:
                        current_mask = torch.ones(pooled.shape[0]).to(device)
                    
                    modality_stats[mod]['valid_samples'] += current_mask.sum().item()

                    # B. Get relevant prototypes for this specific modality
                    is_relevant_mod = is_relevant_global[:, start:end]
                    
                    # C. Count per image. Result is [Batch]
                    ls_per_image_mod = torch.count_nonzero(is_relevant_mod, dim=1).float()
                    
                    # D. Zero out if modality is missing
                    ls_per_image_mod = ls_per_image_mod * current_mask
                    
                    # E. Save to modality statistics
                    modality_stats[mod]['active_protos'] += ls_per_image_mod.sum().item()
                    
                    # F. Add to global sum for the batch
                    batch_ls_global += ls_per_image_mod

            # 3. Update global counter with the sum
            local_size_total_sum_of_mod += batch_ls_global.sum().item()
            
            # Update confusion matrix & preds
            cm_batch = np.zeros((net.module._num_classes, net.module._num_classes), dtype = int)
            for y_pred, y_true in zip(ys_pred, ys):
                cm[y_true][y_pred] += 1
                cm_batch[y_true][y_pred] += 1

            acc = acc_from_cm(cm_batch)

            (top1accs, top3accs) = topk_accuracy(out, ys, topk=[1,3])
            global_top1acc += torch.sum(top1accs).item()
            global_top3acc += torch.sum(top3accs).item()

            y_preds += ys_pred_scores.detach().tolist()     
            y_trues += ys.detach().tolist()
            y_preds_classes += ys_pred.detach().tolist()    
        
        del out, pooled, ys_pred

    print("PIP-Net abstained from a decision for", abstained.item(), "images", flush = True)     
    
    # --- Results Compilation ---
    info['num non-zero prototypes'] = torch.gt(net.module._classification.weight, 1e-3).any(dim = 0).sum().item()
    
    if modality_ranges:
        for mod, (start, end) in modality_ranges.items():
            w_mod = cls_weights[:, start:end]

            # A. Global Non-Zero (Vocabulary - Number of unique prototypes used)
            count_nz = torch.gt(w_mod, 1e-3).any(dim=0).sum().item()
            info[f'num_non_zero_prototypes_{mod}'] = count_nz
            
            # --- Sparsity per modality ---
            total_params_mod = torch.numel(w_mod)
            active_params_mod = torch.count_nonzero(torch.nn.functional.relu(w_mod - 1e-3)).item()
            
            if total_params_mod > 0:
                info[f'sparsity_{mod}'] = (total_params_mod - active_params_mod) / total_params_mod
            else:
                info[f'sparsity_{mod}'] = 0.0
            # -----------------------------------

            # B. Local Size (Average number of RELEVANT prototypes per image)
            total_active = modality_stats[mod]['active_protos']
            total_valid = modality_stats[mod]['valid_samples']
            
            if total_valid > 0:
                info[f'local_size_{mod}'] = total_active / total_valid
            else:
                info[f'local_size_{mod}'] = 0.0

    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    info['top1_accuracy'] = global_top1acc/len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    info['top3_accuracy'] = global_top3acc/len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    info['almost_sim_nonzeros'] = global_sim_anz/len(test_loader.dataset)
    info['local_size_all_classes'] = local_size_total/len(test_loader.dataset)
    info['local_size_sum_of_mod'] = local_size_total_sum_of_mod/len(test_loader.dataset)
    info['almost_nonzeros'] = global_anz/len(test_loader.dataset) # NOTE: These are RAW detections (without weight-filter)

    try:
        y_probs_np = np.array(y_probs_all)
        
        if net.module._num_classes == 2:
            # Binary
            auc_val = roc_auc_score(y_trues, y_probs_np[:, 1])
            info["auc_macro"] = auc_val
            info["auc_class_1"] = auc_val
        else:
            # Multiclass (3+ classes)
            all_classes = list(range(net.module._num_classes))
            
            # 1. Class-specific AUC
            auc_classes = roc_auc_score(y_trues, y_probs_np, multi_class='ovr', average=None, labels=all_classes)
            
            for c, auc_c in enumerate(auc_classes):
                info[f"auc_class_{c}"] = auc_c
                
            # 2. Macro (Unweighted average)
            info["auc_macro"] = np.mean(auc_classes)
            
            # 3. Weighted 
            info["auc_weighted"] = roc_auc_score(y_trues, y_probs_np, multi_class='ovr', average='weighted', labels=all_classes)
            
    except Exception as e:
        print(f"[WARN] Could not calculate AUC: {e}")
        info["auc_macro"] = 0.0

    f1_avg = 'binary' if net.module._num_classes == 2 else 'macro'
    info["f1"] = f1_score(y_trues, y_preds_classes, average=f1_avg)
    
    # Sparsity
    total_params = torch.numel(net.module._classification.weight)
    active_params = torch.count_nonzero(torch.nn.functional.relu(net.module._classification.weight-1e-3)).item()
    info["sparsity"] = (total_params - active_params) / total_params
    info["balanced_accuracy"] = balanced_accuracy_score(y_trues, y_preds_classes)

    sensitivities, specificities = [], []
    for c in range(net.module._num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_c = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        info[f"sensitivity_class_{c}"] = sensitivity
        info[f"specificity_class_{c}"] = specificity
        info[f"f1_class_{c}"] = f1_c
        
    info["macro_sensitivity"] = np.mean(sensitivities)
    info["macro_specificity"] = np.mean(specificities)

    # --- SPECIAL HANDLING FOR BINARY ---
    if net.module._num_classes == 2:
        # Assumption: Class 1 is "positive" (Disease/AD)
        # Sensitivity = How well do we find Class 1?
        info["sensitivity"] = info["sensitivity_class_1"]
        
        # Specificity = How well do we find Class 0 (i.e. reject disease)?
        # Mathematically equivalent to sensitivity_class_0
        info["specificity"] = info["sensitivity_class_0"]

    return info


def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total


@torch.no_grad()                    
def get_local_explanations(
        net, 
        projectloader, 
        device,
        args: argparse.Namespace,
        plot = False,
        plot_limit_per_modality = 20,
        max_samples = None,
        threshold = None
        ):
    
    print("Detect prototypes in predictions...", flush = True)

    dir = os.path.join(args.log_dir, "clinical_feedback_local_explanations")
    if plot:
        if not os.path.exists(dir):
            os.makedirs(dir)
        plot_dir = os.path.join(dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    local_explanations = []
    y_preds = []
    y_trues = []
    
    try:
        class_to_idx = projectloader.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    except AttributeError:
        idx_to_class = {}

    dataset_paths = projectloader.dataset.X_paths
    
    net.eval()
    classification_weights = net.module._classification.weight

    modalities = net.module.modalities 
    modality_offsets = {}
    current_offset = 0
    
    # Initialize counter for combinations (for plotting)
    plots_count_combinations = {} 
    
    for mod in modalities:
        add_on_module = net.module._add_ons[mod]
        num_protos = 0
        if hasattr(add_on_module, '_num_prototypes'):
             num_protos = add_on_module._num_prototypes
        else:
            for m in add_on_module.modules():
                if isinstance(m, torch.nn.Conv3d):
                    num_protos = m.out_channels
                    break
        
        if num_protos == 0:
            num_protos = getattr(args, 'num_features', 512)
            if num_protos == 0: num_protos = 512 

        modality_offsets[mod] = (current_offset, current_offset + num_protos)
        current_offset += num_protos

    img_iter = tqdm(enumerate(projectloader), total=len(projectloader), desc="Processing images", mininterval=5., ncols=0)
    
    for k, (xs, ms, ys) in img_iter: 
        
        if max_samples is not None and k >= max_samples:
            print(f"Reached max_samples ({max_samples}). Stopping.")
            break

        ys = ys.to(device)
        xs = {key: val.to(device) for key, val in xs.items()}
        ms = {key: val.to(device) for key, val in ms.items()} if ms is not None else None
        
        img_name_ref = "Unknown_Ref"
        for mod in modalities:
            path = dataset_paths[mod][k]
            if not pd.isna(path) and str(path).lower() != 'nan':
                img_name_ref = str(path)
                break

        local_explanation = dict() 
        cached_images = {} 

        with torch.no_grad():
            softmaxes_dict, pooled, out = net(xs, masks=ms, inference=True, threshold=threshold) 
            
            max_out_score, ys_pred = torch.max(out, dim=1) 
            y_preds.append(ys_pred.item())
            y_trues.append(ys.item())
            
            pred_class_idx = ys_pred.item()
            
            sorted_pooled, sorted_pooled_indices = torch.sort(pooled.squeeze(0), descending=True) 
            
            for prototype_idx in sorted_pooled_indices:
                p_idx_item = prototype_idx.item()
                
                simweight = pooled[0, prototype_idx].item() * net.module._classification.weight[pred_class_idx, prototype_idx].item()
                
                if abs(simweight) > 0.01:
                    c_weight = torch.max(classification_weights[:, prototype_idx]) 
                        
                    if (c_weight > 1e-10):
                    
                        target_mod = None
                        local_p_idx = 0
                        for mod, (start, end) in modality_offsets.items():
                            if start <= p_idx_item < end:
                                target_mod = mod
                                local_p_idx = p_idx_item - start
                                break
                        
                        if target_mod is None: continue

                        softmax_map = softmaxes_dict[target_mod]
                        max_hw, max_idx_hw = torch.max(softmax_map[0, local_p_idx, :, :, :], dim=0) 
                        max_h, max_idx_h = torch.max(max_hw, dim=0) 
                        max_w, max_idx_w = torch.max(max_h, dim=0)  

                        w_idx = max_idx_w.item()
                        h_idx = max_idx_h[w_idx].item()
                        d_idx = max_idx_hw[h_idx, w_idx].item()
                        
                        if target_mod not in cached_images:
                            if target_mod in xs:
                                img_tensor = xs[target_mod].cpu()
                                cached_images[target_mod] = img_tensor
                            else:
                                cached_images[target_mod] = None

                        img_tensor = cached_images[target_mod]
                        
                        if img_tensor is not None:
                            current_shape = img_tensor.shape[-3:] # (D, H, W)
                            
                            patchsize, skip_z, skip_y, skip_x = get_patch_size_dynamic(current_shape, args)
                            
                            ps_coord = get_img_coordinates(
                                current_shape[0], current_shape[1], current_shape[2], 
                                softmax_map.shape, 
                                patchsize, skip_z, skip_y, skip_x,
                                d_idx, h_idx, w_idx)
                            
                            local_explanation[p_idx_item] = (ps_coord, simweight)
                            
        local_explanations.append(local_explanation)
        
        # --- PLOTTING LOGIC (Sorted per True Class + Title/Footer) ---
        if plot and len(local_explanation) > 0:
            
            contributing_modalities = set()
            for proto_idx in local_explanation.keys():
                for mod, (start, end) in modality_offsets.items():
                    if start <= proto_idx < end:
                        contributing_modalities.add(mod)
                        break
            
            if len(contributing_modalities) > 0:
                true_idx = ys.item()
                true_name = idx_to_class.get(true_idx, str(true_idx))
                
                pred_idx = ys_pred.item()
                pred_name = idx_to_class.get(pred_idx, str(pred_idx))

                combo_str = "_".join(sorted(list(contributing_modalities)))
                plot_counter_key = f"{true_name}_{combo_str}"
                
                if plot_counter_key not in plots_count_combinations:
                    plots_count_combinations[plot_counter_key] = 0
                
                if plots_count_combinations[plot_counter_key] < plot_limit_per_modality:
                    proto_list = sorted(list(local_explanation.keys()))
                    proto_str = str(proto_list)
                    wrapped_proto_str = textwrap.fill(proto_str, width=80, subsequent_indent=" "*13)

                    main_title = f"True: {true_name} | Pred: {pred_name}"
                    footer_text = f"Detected PS: {wrapped_proto_str}"

                    try:
                        img_str = str(img_name_ref)
                        match_subj = re.search(r"(\d{3}_S_\d{4})", img_str)
                        subj = match_subj.group(1) if match_subj else "UnknownSubj"
                        base_name = os.path.basename(img_str)
                        exam, _ = os.path.splitext(base_name)

                        # Create class directory (e.g., True_AD)
                        class_plot_dir = os.path.join(plot_dir, f"True_{true_name}")
                        if not os.path.exists(class_plot_dir):
                            os.makedirs(class_plot_dir)

                        ps_name = f"local_expl_{true_name}_{combo_str}_{subj}_{exam}_idx{k}"
                        plot_name = os.path.join(class_plot_dir, ps_name + ".png")
                        
                        xs_cpu = {key: val.cpu() for key, val in xs.items()}
                        
                        plot_local_explanation(
                            xs_cpu, 
                            local_explanation, 
                            modality_offsets, 
                            title=main_title, 
                            footer=footer_text, 
                            save_path=plot_name
                        )
                        
                        plots_count_combinations[plot_counter_key] += 1

                    except Exception as e:
                        print(f"[WARN] Could not plot explanation: {e}")
        # ---------------------------------------------------------
        
    return local_explanations, y_preds, y_trues
                        

@torch.no_grad()
def eval_local_explanations(
        net, 
        local_explanations, 
        device,
        args: argparse.Namespace):
    
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    relevant_ps = [ps.item() for ps in classification_weights.nonzero(as_tuple=True)[1]] 
    
    ps_cc_coords = {ps:[] for ps in relevant_ps}
    ps_scores = {ps:[] for ps in relevant_ps}
    
    for i, local_explanation in tqdm(enumerate(local_explanations), total=len(local_explanations), desc="Processing explanations", mininterval=5., ncols=0):
        
        proto_found = list(local_explanation.keys())
        proto_found.sort()
        proto_not_found = list(set(relevant_ps)-set(local_explanation.keys()))
        proto_not_found.sort()
        
        for ps in proto_found:
            dmin, dmax, hmin, hmax, wmin, wmax = local_explanation[ps][0]
            ps_cc_d = (dmin + dmax)/2
            ps_cc_h = (hmin + hmax)/2
            ps_cc_w = (wmin + wmax)/2
            ps_cc = np.array([ps_cc_d, ps_cc_h, ps_cc_w])
            ps_cc_coords[ps].append(ps_cc)
            ps_scores[ps].append(local_explanation[ps][1])
        
        for ps in proto_not_found:
            ps_cc_coords[ps].append(None)
            ps_scores[ps].append(None)
    
    ps_scores_df = pd.DataFrame(ps_scores)
    
    # How many times each prototype is detected dataset
    ps_detections = ps_scores_df.count()
    
    # Variation in prototypes' localization
    mean_ps_coords = dict()
    std_ps_coords = dict()
    lc_ps = dict() # ps localizations' consistency
    
    for ps in relevant_ps: 
        
        valid_coords = [ps_coord for ps_coord in ps_cc_coords[ps] if ps_coord is not None]
        if len(valid_coords) > 0:
            mean_ps_coords[ps] = np.array(valid_coords).mean(0) # mean of ps coordinates 
            std_ps_coords[ps] = np.array(valid_coords).std(0) # std of ps' coordinates
            lc_ps[ps] = np.array([np.linalg.norm(ps_coord-mean_ps_coords[ps])/(np.sqrt(3)*32) for ps_coord in valid_coords]).mean(0)
        else:
            mean_ps_coords[ps] = np.zeros(3)
            std_ps_coords[ps] = np.zeros(3)
            lc_ps[ps] = 0.0
            
    return ps_detections, mean_ps_coords, std_ps_coords, lc_ps


@torch.no_grad()
def check_empty_prototypes(args, net, img_prototype_top1, proto_coord_top1):
    
    empty_ps = []
    return empty_ps
                

@torch.no_grad()
def get_thresholds(net,
        test_loader: DataLoader,
        epoch,
        device,
        percentile: float = 95.,
        log: Log = None,  
        log_prefix: str = 'log_eval_epochs', 
        progress_prefix: str = 'Get Thresholds Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
    
    outputs_per_class = dict()
    outputs_per_correct_class = dict()
    for c in range(net.module._num_classes):
        outputs_per_class[c] = []
        outputs_per_correct_class[c] = []
        
    # Show progress on progress bar
    test_iter = iter(test_loader)
    
    # Iterate through the test set
    for i, (xs, ms, ys) in enumerate(test_iter):
        ys = ys.to(device)
        xs = {k: v.to(device) for k, v in xs.items()}
        ms = {k: v.to(device) for k, v in ms.items()} if ms is not None else None
        
        with torch.no_grad():
            _, pooled, out = net(xs, masks=ms)

            ys_pred = torch.argmax(out, dim=1)
            for pred in range(len(ys_pred)):
                outputs_per_class[
                    ys_pred[pred].item()].append(out[pred,:].max().item())
                
                if ys_pred[pred].item()==ys[pred].item():
                    outputs_per_correct_class[
                        ys_pred[pred].item()].append(out[pred,:].max().item())
        
        del out
        del pooled
        del ys_pred

    class_thresholds = dict()
    correct_class_thresholds = dict()
    all_outputs = []
    all_correct_outputs = []
    
    for c in range(net.module._num_classes):
        if len(outputs_per_class[c])>0:
            outputs_c = outputs_per_class[c]
            all_outputs += outputs_c
            class_thresholds[c] = np.percentile(outputs_c, 100-percentile) 
            
        if len(outputs_per_correct_class[c])>0:
            correct_outputs_c = outputs_per_correct_class[c]
            all_correct_outputs += correct_outputs_c
            correct_class_thresholds[c] = np.percentile(correct_outputs_c, 100-percentile)
    
    overall_threshold = np.percentile(all_outputs, 100-percentile)
    overall_correct_threshold = np.percentile(all_correct_outputs, 100-percentile)
    
    mean_ct = np.mean(list(class_thresholds.values())) if class_thresholds else 0
    mean_cct = np.mean(list(correct_class_thresholds.values())) if correct_class_thresholds else 0
    
    for c in range(net.module._num_classes):
        
        if c not in class_thresholds.keys():
            class_thresholds[c] = mean_ct
            
        if c not in correct_class_thresholds.keys():
            correct_class_thresholds[c] = mean_cct

    calculated_percentile = 0
    correctly_classified = 0
    total = 0
    
    for c in range(net.module._num_classes):
        correctly_classified+=sum(i>class_thresholds[c] for i in outputs_per_class[c])
        total += len(outputs_per_class[c])
        
    if total > 0:
        calculated_percentile = correctly_classified/total

    if percentile < 100:
        
        while calculated_percentile < (percentile/100.):
            class_thresholds.update(
                (x, y*0.999) for x, y in class_thresholds.items())
            correctly_classified = 0
            
            for c in range(net.module._num_classes):
                correctly_classified+=sum(i>=class_thresholds[c] for i in outputs_per_class[c])
            if total > 0:
                calculated_percentile = correctly_classified/total
            else:
                break

    return overall_correct_threshold, overall_threshold, correct_class_thresholds, class_thresholds


@torch.no_grad()
def eval_ood(net,
        test_loader: DataLoader,
        epoch,
        device,
        threshold, 
        progress_prefix: str = 'Get Thresholds Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
 
    predicted_as_id = 0
    seen = 0.
    abstained = 0

    test_iter = iter(test_loader)
    
    # Iterate through the test set
    for i, (xs, ms, ys) in enumerate(test_iter):
        ys = ys.to(device)
        xs = {k: v.to(device) for k, v in xs.items()}
        ms = {k: v.to(device) for k, v in ms.items()} if ms is not None else None
        
        with torch.no_grad():
            _, pooled, out = net(xs, masks=ms)
            max_out_score, ys_pred = torch.max(out, dim=1)
            ys_pred = torch.argmax(out, dim=1)
            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))
            
            for j in range(len(ys_pred)):
                seen+=1.
                if isinstance(threshold, dict):
                    thresholdj = threshold[ys_pred[j].item()]
                elif isinstance(threshold, float): #overall threshold
                    thresholdj = threshold
                else:
                    raise ValueError("provided threshold should be float or dict", type(threshold))
                sample_out = out[j,:]
                
                if sample_out.max().item() >= thresholdj:
                    predicted_as_id += 1
                    
            del out
            del pooled
            del ys_pred
            
    print("Samples seen:", seen, "of which predicted as In-Distribution:", predicted_as_id, flush=True)
    print("PIP-Net abstained from a decision for", abstained.item(), "images", flush=True)
    
    return predicted_as_id/seen