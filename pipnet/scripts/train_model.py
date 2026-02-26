import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_pipnet(
        net, 
        train_loader, 
        optimizer_net, 
        optimizer_classifier, 
        scheduler_net, 
        scheduler_classifier, 
        criterion, 
        epoch, 
        nr_epochs, 
        device, 
        pretrain = False, 
        finetune = False, 
        progress_prefix: str = 'Train Epoch',
        mask = None): # Mask is a Dict: {'mri': tensor, 'amy': tensor}

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix+'%s'%epoch, mininterval=2., ncols=0)
    
    # Count parameters (info only)
    count_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1  
    
    # Print only once at the beginning of the epoch
    if epoch == 0 or epoch == 1:
        print("Number of parameters that require gradient: ", count_param, flush=True)

    # Weights configuration
    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        cl_weight = 2.
    
    unif_weight = 0. 

    lrs_net = []
    lrs_class = []

    # Cache for downsampled global masks
    mask_downsampled = {} 

    # --- START TRAINING LOOP ---
    for i, (inputs1, inputs2, presence_masks, ys) in train_iter:       
        
        # 1. Move data to Device
        ys = ys.to(device)
        
        for k in inputs1: inputs1[k] = inputs1[k].to(device)
        for k in inputs2: inputs2[k] = inputs2[k].to(device)
        for k in presence_masks: presence_masks[k] = presence_masks[k].to(device)
        
        # 2. Concatenate View 1 and View 2 (Batch Size is doubled)
        inputs_concat = {}
        for mod in inputs1.keys():
            inputs_concat[mod] = torch.cat([inputs1[mod], inputs2[mod]], dim=0)

        # IMPORTANT: Also double the masks since we have twice the number of images now
        presence_masks_concat = {}
        for mod in presence_masks.keys():
            presence_masks_concat[mod] = torch.cat([presence_masks[mod], presence_masks[mod]], dim=0)

        # 3. Reset gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # 4. Forward Pass
        # Net handles missing modalities internally if presence_masks are provided
        proto_features_dict, pooled, out = net(inputs_concat, masks=presence_masks_concat)

        # 5. --- GLOBAL ANATOMICAL MASKING ---
        if mask is not None:
            for mod, global_m in mask.items():
                # Check if modality exists in features AND if not already cached
                if mod in proto_features_dict and mod not in mask_downsampled:
                    if global_m is not None:
                        # Get size from features (BS, NumProtos, D, H, W)
                        target_size = proto_features_dict[mod].shape[-3:] 
                        
                        temp_mask = global_m
                        # Ensure dimensions are correct for interpolate (1, 1, D, H, W)
                        while temp_mask.ndim < 5:
                            temp_mask = temp_mask.unsqueeze(0)
                            
                        # Downsample nearest neighbor
                        downsampled = F.interpolate(temp_mask, size=target_size, mode='nearest')
                        mask_downsampled[mod] = (downsampled > 0.5).float().to(device)
            
            # Apply the masks
            for mod in proto_features_dict.keys():
                if mod in mask_downsampled:
                    # Multiply features with the mask (zeros out the background)
                    proto_features_dict[mod] = proto_features_dict[mod] * mask_downsampled[mod]
        
        # 6. Calculate Loss
        loss, acc = calculate_loss(
            proto_features_dict, 
            pooled, 
            out, 
            ys, 
            align_pf_weight, 
            t_weight, 
            unif_weight, 
            cl_weight, 
            net.module._classification.normalization_multiplier, 
            pretrain, 
            finetune, 
            criterion, 
            train_iter, 
            presence_masks=presence_masks,
            print = True, 
            EPS = 1e-8)
        
        # 7. Backward & Step
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        # 8. Clamp weights (Non-negative constraint for PIPNet)
        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) 
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp( net.module._classification.bias.data, min=0.))  
                    
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info


def calculate_loss(
        proto_features_dict,
        pooled, 
        out, 
        ys1, 
        align_pf_weight, 
        t_weight, 
        unif_weight, 
        cl_weight, 
        net_normalization_multiplier, 
        pretrain, 
        finetune, 
        criterion, 
        train_iter, 
        presence_masks=None, 
        print = True, 
        EPS = 1e-10):
    
    # Labels duplicated (View 1 + View 2)
    ys = torch.cat([ys1, ys1])
    
    # 1. Tanh Loss (Sparsity)
    pooled1, pooled2 = pooled.chunk(2) 
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean())/2.

    # 2. Alignment Loss (Per Modality)
    a_loss_pf = 0.
    num_active_modalities = 0
    
    for mod in proto_features_dict.keys():
        pf_mod = proto_features_dict[mod]
        
        # Split into View 1 and View 2
        pf1, pf2 = pf_mod.chunk(2)
        
        # Flatten spatial dimensions: (B, NumProtos, D, H, W) -> (B, NumProtos, D*H*W) -> (B*D*H*W, NumProtos)
        # Note: Flatten(end_dim=1) merges Batch and Spatial dimensions in dim 0.
        embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        
        # --- ROBUST ALIGNMENT LOSS (Handles missing data) ---
        if presence_masks is not None and mod in presence_masks:
            # Get the mask [Batch]. reshape(-1) ensures it's a 1D vector even if Batch=1
            mask = presence_masks[mod].reshape(-1) 
            
            # If no one in the batch has this modality, skip
            if mask.sum() == 0:
                continue

            # Calculate loss per sample/voxel
            # l1 form: [Batch * SpatialPixels]
            l1 = align_loss_per_sample(embv1, embv2.detach(), EPS)
            l2 = align_loss_per_sample(embv2, embv1.detach(), EPS)
            
            # --- FIX: Expand the mask to match spatial dimensions ---
            # l1.shape[0] is (Batch * Voxels). mask.shape[0] is (Batch).
            if l1.shape[0] != mask.shape[0]:
                # Calculate how many voxels each image has
                num_spatial_pixels = l1.shape[0] // mask.shape[0]
                # Repeat the mask: [1, 0] -> [1, 1, 1... , 0, 0, 0...]
                mask_expanded = mask.repeat_interleave(num_spatial_pixels)
            else:
                mask_expanded = mask

            # Zero out loss for voxels belonging to missing images
            l1 = l1 * mask_expanded
            l2 = l2 * mask_expanded
            
            # Average over only the existing pixels
            # We divide by the sum of mask_expanded (total number of valid voxels)
            mean_l1 = l1.sum() / (mask_expanded.sum() + EPS)
            mean_l2 = l2.sum() / (mask_expanded.sum() + EPS)
            
            mod_loss = (mean_l1 + mean_l2) / 2.
        else:
            # Fallback
            mod_loss = (align_loss(embv1, embv2.detach(), EPS) + align_loss(embv2, embv1.detach(), EPS))/2.
            
        a_loss_pf += mod_loss
        num_active_modalities += 1
    
    # Normalize loss over the number of modalities
    if num_active_modalities > 0:
        a_loss_pf = a_loss_pf / num_active_modalities

    # 3. Total Loss Calculation
    loss = 0.
    if not finetune:
        loss = align_pf_weight * a_loss_pf
        loss += t_weight * tanh_loss
    
    # 4. Classification Loss
    class_loss = torch.tensor(0.).to(ys.device)

    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs), dim=1), ys)
        
        if finetune:
            loss = cl_weight * class_loss
        else:
            loss += cl_weight * class_loss

    # 5. Accuracy Calculation
    acc = 0.
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
    
    # 6. Logging
    if print: 
        with torch.no_grad():
            num_scores_pos = torch.count_nonzero(torch.relu(pooled-0.1), dim=1).float().mean().item()
            
            if pretrain:
                train_iter.set_postfix_str(f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, P>0.1:{num_scores_pos:.1f}', refresh=False)
            else:
                train_iter.set_postfix_str(f'L:{loss.item():.3f}, LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, P>0.1:{num_scores_pos:.1f}, Ac:{acc:.3f}', refresh=False)            

    return loss, acc


def align_loss(inputs, targets, EPS=1e-12):
    """Original (scalar) alignment loss used as fallback."""
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def align_loss_per_sample(inputs, targets, EPS=1e-12):
    """
    Calculates alignment loss but returns a vector [Batch_Size] 
    instead of a scalar mean. This allows for masking.
    """
    assert inputs.shape == targets.shape
    # Dot product per sample
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    # Log loss per sample
    loss = -torch.log(loss + EPS)
    return loss