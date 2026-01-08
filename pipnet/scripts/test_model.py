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
from vis_pipnet import get_patch_size, get_img_coordinates, plot_local_explanation
from plot_utils import plot_3d_slices, plot_rgb_slices, generate_rgb_array

    

@torch.no_grad()
def eval_pipnet(
        net,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,  
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()
    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    cm = np.zeros((net.module._num_classes, net.module._num_classes), dtype = int)

    global_top1acc = 0.
    global_top3acc = 0.
    global_sim_anz = 0.
    global_anz = 0.
    local_size_total = 0.
    y_trues = []
    y_preds = []
    y_preds_classes = []
    abstained = 0
    
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc=progress_prefix+' %s'%epoch, mininterval=5., ncols=0)
    
    # Iterate through the test set
    for i, (xs, ms, ys) in test_iter:
        
        # Move inputs to device. xs and ms are dictionaries of tensors.
        ys = ys.to(device)
        xs = {k: v.to(device) for k, v in xs.items()}
        ms = {k: v.to(device) for k, v in ms.items()} if ms is not None else None
        
        with torch.no_grad():

            # Use the model to classify this batch of input data
            _, pooled, out = net(xs, masks=ms, inference = True)
            
            max_out_score, ys_pred = torch.max(out, dim=1) # max, max_idx
            ys_pred_scores = torch.amax(F.softmax((torch.log1p(out**net.module._classification.normalization_multiplier)), dim = 1), dim = 1) # class confidence scores
            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))  
            repeated_weight = net.module._classification.weight.unsqueeze(1).repeat(1, pooled.shape[0], 1)
            sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(pooled*repeated_weight), 1e-3).float(), dim = 2).float()
            local_size = torch.count_nonzero(torch.gt(torch.relu((pooled*repeated_weight) - 1e-3).sum(dim = 1), 0.).float(), dim = 1).float()
            local_size_total += local_size.sum().item()
            correct_class_sim_scores_anz = torch.diagonal(torch.index_select(sim_scores_anz, dim = 0, index = ys_pred), 0)
            global_sim_anz += correct_class_sim_scores_anz.sum().item()
            almost_nz = torch.count_nonzero(torch.gt(torch.abs(pooled), 1e-3).float(), dim = 1).float()
            global_anz += almost_nz.sum().item()
            # Update the confusion matrix
            cm_batch = np.zeros((net.module._num_classes, net.module._num_classes), dtype = int)
            
            for y_pred, y_true in zip(ys_pred, ys):
                
                cm[y_true][y_pred] += 1
                cm_batch[y_true][y_pred] += 1
                
            acc = acc_from_cm(cm_batch)   

            (top1accs, top3accs) = topk_accuracy(out, ys, topk=[1,3])
            
            global_top1acc += torch.sum(top1accs).item()
            global_top3acc += torch.sum(top3accs).item()
            y_preds += ys_pred_scores.detach().tolist()     # predicted class' confidence scores
            y_trues += ys.detach().tolist()
            y_preds_classes += ys_pred.detach().tolist()    # predicted classes
        
        del out
        del pooled
        del ys_pred
        
    print("PIP-Net abstained from a decision for", abstained.item(), "images", flush = True)     
       
    info['num non-zero prototypes'] = torch.gt(net.module._classification.weight, 1e-3).any(dim = 0).sum().item()
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    info['top1_accuracy'] = global_top1acc/len(test_loader.dataset)
    info['top3_accuracy'] = global_top3acc/len(test_loader.dataset)
    info['almost_sim_nonzeros'] = global_sim_anz/len(test_loader.dataset)
    info['local_size_all_classes'] = local_size_total/len(test_loader.dataset)
    info['almost_nonzeros'] = global_anz/len(test_loader.dataset)
    # f1 = 'binary' if net.module._num_classes == 2 else 'weighted'
    # info["f1"] = f1_score(y_trues, y_preds_classes, average=f1)
    f1_avg = 'binary' if net.module._num_classes == 2 else 'macro'
    info["f1"] = f1_score(y_trues, y_preds_classes, average=f1_avg)
    info["sparsity"] = (torch.numel(net.module._classification.weight) - torch.count_nonzero(torch.nn.functional.relu(net.module._classification.weight-1e-3)).item()) / torch.numel(net.module._classification.weight)
    info["balanced_accuracy"] = balanced_accuracy_score(y_trues, y_preds_classes)

    # --- HÄR BÖRJAR DEN NYA LOGIKEN FÖR KLASS-SPECIFIKA MÅTT ---
    
    # Hämta klassnamn om de finns tillgängliga, annars använd index 0,1,2
    # Antag att class_names = ['CN', 'MCI', 'AD'] eller liknande
    # Du kan hämta detta från test_loader.dataset.classes om det finns
    
    if net.module._num_classes == 2:
        # Din befintliga kod för binär
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        info["sensitivity"] = tp/(tp+fn) if (tp+fn) > 0 else 0
        info["specificity"] = tn/(tn+fp) if (tn+fp) > 0 else 0
        
        # print(f"\nEpoch {epoch} (Binary): TP:{tp} FN:{fn} FP:{fp} TN:{tn}", flush=True)

    else:
        # LOGIK FÖR 3 KLASSER (One-vs-Rest)
        # Vi loopar igenom varje klass och ser den som "Positiv" och resten som "Negativa"
        
        # print(f"\nEpoch {epoch} (Multi-class breakdown):", flush=True)
        
        sensitivities = []
        specificities = []
        
        # cm rader = True, kolumner = Pred
        for c in range(net.module._num_classes):
            # True Positive (Diagonalen)
            tp = cm[c, c]
            
            # False Negative (Summan av raden c, minus TP) -> Missade fall av klass c
            fn = np.sum(cm[c, :]) - tp
            
            # False Positive (Summan av kolumn c, minus TP) -> Felaktigt gissat på klass c
            fp = np.sum(cm[:, c]) - tp
            
            # True Negative (Totala summan minus rad c och kolumn c)
            tn = np.sum(cm) - (tp + fp + fn)
            
            # Beräkna mått
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_c = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            
            # Spara i info-dicten. Exempel: "sensitivity_0" (för CN), "sensitivity_1" (för MCI)
            info[f"sensitivity_class_{c}"] = sensitivity
            info[f"specificity_class_{c}"] = specificity
            info[f"f1_class_{c}"] = f1_c
            
            # print(f" Class {c}: Sens: {sensitivity:.3f}, Spec: {specificity:.3f}, F1: {f1_c:.3f}", flush=True)

        # Macro-average (verifiering av balanced acc och macro-f1)
        info["macro_sensitivity"] = np.mean(sensitivities) # Samma som Balanced Acc
        info["macro_specificity"] = np.mean(specificities)
        
        # Top-3 är meningslöst för 3 klasser (alltid 1.0), ta bort eller sätt till acc
        # info['top3_accuracy'] = info['test_accuracy'] 

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
        plot = False):
    """
    Compute local explanations for Multimodal PIPNet.
    Handles dictionary inputs/outputs and maps global prototype indices
    to specific modalities.
    """
    
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
    
    # --- NY KOD START: Skapa översättningstabell (0 -> 'CN', 1 -> 'AD') ---
    # Vi hämtar class_to_idx från datasetet och vänder på den
    try:
        class_to_idx = projectloader.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"[INFO] Class mapping found: {class_to_idx}")
    except AttributeError:
        print("[WARN] Could not find class_to_idx in dataset. Using numbers.")
        idx_to_class = {} # Fallback
    # --- NY KOD SLUT ---
    
    patchsize, skip_z, skip_y, skip_x = get_patch_size(args)

    dataset_paths = projectloader.dataset.X_paths
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight

    modalities = net.module.modalities 
    modality_offsets = {}
    current_offset = 0
    
    for mod in modalities:
        # --- FIX: Robust check for number of prototypes ---
        add_on_module = net.module._add_ons[mod]
        num_protos = 0
        
        # Check attribute or look for Conv3d recursively
        if hasattr(add_on_module, '_num_prototypes'):
             num_protos = add_on_module._num_prototypes
        else:
            for m in add_on_module.modules():
                if isinstance(m, torch.nn.Conv3d):
                    num_protos = m.out_channels
                    break
        
        # Fallback
        if num_protos == 0:
            num_protos = getattr(args, 'num_features', 512)
            # Dubbelkolla att vi inte fick 0 från args heller
            if num_protos == 0: 
                num_protos = 512 # Hårdkodad sista utväg
        # --------------------------------------------------

        modality_offsets[mod] = (current_offset, current_offset + num_protos)
        current_offset += num_protos

    img_iter = tqdm(enumerate(projectloader), total=len(projectloader), desc="Processing images", mininterval=5., ncols=0)
    
    for k, (xs, ms, ys) in img_iter: 
        
        ys = ys.to(device)
        xs = {key: val.to(device) for key, val in xs.items()}
        ms = {key: val.to(device) for key, val in ms.items()} if ms is not None else None
        
        # --- FIX: Hitta en giltig sökväg från VILKEN modalitet som helst ---
        img_name_ref = "Unknown_Ref"
        for mod in modalities:
            path = dataset_paths[mod][k]
            if not pd.isna(path) and str(path).lower() != 'nan':
                img_name_ref = str(path)
                break

        local_explanation = dict() 
        
        with torch.no_grad():
            
            softmaxes_dict, pooled, out = net(xs, masks=ms, inference=True) 
            
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0), descending=True) 
            max_out_score, ys_pred = torch.max(out, dim=1) 
            
            y_preds.append(ys_pred.item())
            y_trues.append(ys.item())
            
            for pred_class_idx in sorted_out_indices:
                
                sorted_pooled, sorted_pooled_indices = torch.sort(pooled.squeeze(0), descending=True) 
                
                simweights = []
                
                for prototype_idx in sorted_pooled_indices:
                    
                    p_idx_item = prototype_idx.item()
                    simweight = pooled[0, prototype_idx].item() * net.module._classification.weight[pred_class_idx, prototype_idx].item()
                    
                    simweights.append(simweight)
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
                            
                            if target_mod is None:
                                continue

                            softmax_map = softmaxes_dict[target_mod]

                            # get the coordinate of the maximum in the feature's space
                            max_hw, max_idx_hw = torch.max(softmax_map[0, local_p_idx, :, :, :], dim=0) 
                            max_h, max_idx_h = torch.max(max_hw, dim=0) 
                            max_w, max_idx_w = torch.max(max_h, dim=0)  

                            w_idx = max_idx_w.item()
                            h_idx = max_idx_h[w_idx].item()
                            d_idx = max_idx_hw[h_idx, w_idx].item()
                            
                            img_path = dataset_paths[target_mod][k]
                            
                            if pd.isna(img_path) or str(img_path).lower() == 'nan':
                                continue
                                
                            img_np = np.load(img_path).astype(np.float32)
                            
                            # Handle channels
                            if img_np.ndim == 3:
                                img_np = np.expand_dims(img_np, axis=0) 
                            elif img_np.ndim == 4:
                                img_np = np.transpose(img_np, (3, 0, 1, 2)) 
                            
                            img_tensor = torch.from_numpy(img_np)
                            img_tensor = transforms.Resize(spatial_size = (args.slices, args.rows, args.cols))(img_tensor)
                            img_tensor = img_tensor.unsqueeze(0) 
                            
                            ps_coord = get_img_coordinates(
                                args.slices, args.rows, args.cols, 
                                softmax_map.shape, 
                                patchsize, skip_z, skip_y, skip_x,
                                d_idx, h_idx, w_idx)
                            
                            local_explanation[p_idx_item] = (ps_coord, simweight)
                            
        local_explanations.append(local_explanation)
        # title = "Prediction " + str(ys_pred.item()) + "\n Detected PS: " + str(list(local_explanation.keys()))
        
        # --- NY KOD START: Hämta det läsbara namnet ---
        # ys_pred är indexet (t.ex. 1). Vi slår upp det i idx_to_class (t.ex. 'AD').
        pred_idx = ys_pred.item()
        pred_name = idx_to_class.get(pred_idx, str(pred_idx)) # Fallback till siffra om namn saknas
        
        # Samma sak för sanningen (ys) om du vill visa det också
        true_idx = ys.item()
        true_name = idx_to_class.get(true_idx, str(true_idx))

        title = f"Pred: {pred_name} | True: {true_name}\n Detected PS: {str(list(local_explanation.keys()))}"
        # --- NY KOD SLUT ---

        if plot:
            try:
                img_str = str(img_name_ref)
                
                # --- FIX: Robust Regex och Basename ---
                # 1. Hitta Subjekt ID (XXX_S_XXXX) oavsett path-struktur
                match_subj = re.search(r"(\d{3}_S_\d{4})", img_str)
                if match_subj:
                    subj = match_subj.group(1)
                else:
                    subj = "UnknownSubj"
                
                # 2. Hitta Filnamn/Exam ID (ta bort .npy och path)
                # Detta funkar både för /.../uuid.npy och /.../I12345.npy
                base_name = os.path.basename(img_str)
                exam, _ = os.path.splitext(base_name)
                # --------------------------------------

                text = "local_expl_"
                ps_name = text + subj + "_" + exam + "_idx" + str(k)
                plot_name = os.path.join(plot_dir, ps_name + ".png")
                
                xs_cpu = {k: v.cpu() for k, v in xs.items()}
                
                # Skicka med modality_offsets!
                plot_local_explanation(xs_cpu, local_explanation, modality_offsets, title=title, save_path=plot_name)
                
            except Exception as e:
                print(f"[WARN] Could not plot explanation for {img_name_ref}: {e}")
        
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
    # Placeholder: Skipping check if not fully adapted to MM
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

