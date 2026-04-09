import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PrefetchDataset
from sklearn.metrics import *
from loss import MaskedBCEWithLogitsLoss

def validate_model(model, config, val_files, criterion, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    torch.set_num_threads(40)
    model.to(device)
    model.eval()
    criterion = MaskedBCEWithLogitsLoss()
    y_true, y_probs = [], []
    num_batches = 0
    val_loss = 0
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    dataset_val = PrefetchDataset(val_files, batch_size, seq_len, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, method='minmax', transform=True, shuffle=False, random_start=False, device=device)

    loader_val = DataLoader(dataset_val, batch_size=None, num_workers=0) 
    with torch.no_grad():
        for batch in loader_val:
            inputs = batch['input'].to(device, non_blocking=True)  
            labels = batch['label'].view(-1, batch['label'].shape[-1]).to(device, non_blocking=True)   
            mask = batch['mask'].to(device, non_blocking=True)   
            output = model(inputs, mask)
            val_loss += criterion(output, labels).item()
            probs = torch.sigmoid(output)                
            y_true.append(labels.cpu())
            y_probs.append(probs.cpu())
            num_batches += 1
    
    val_loss /= num_batches
       
    y_true = torch.cat(y_true, dim=0)
    y_probs = torch.cat(y_probs, dim=0)
    mask_lbl = ~(y_true == -100).all(dim=1)
    y_true = y_true[mask_lbl].numpy()
    y_probs = y_probs[mask_lbl].numpy()
    thresholds = find_best_thresholds(y_true, y_probs)
    y_pred = (y_probs >= thresholds).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "F1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "AUC_micro": roc_auc_score(y_true, y_pred, average='micro')
    }

    return metrics, val_loss, thresholds, num_batches

def validate_model_sequential(model, config, val_files, criterion, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    torch.set_num_threads(40)
    model.to(device)
    model.eval()
    criterion = MaskedBCEWithLogitsLoss()
    y_true, y_probs = [], []
    num_batches = 0
    val_loss = 0
    seq_len = config["seq_len"]
    batch_size = 1 
    dataset_val = PrefetchDataset(val_files, batch_size, seq_len, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, method='minmax', transform=True, shuffle=False, random_start=False, 
                              device=device, sequential=True)

    loader_val = DataLoader(dataset_val, batch_size=None, num_workers=0) 
    with torch.no_grad():
        for batch in loader_val:
            inputs = batch['input'].to(device, non_blocking=True)   # (B, seq_len, input_dim)
            labels = batch['label'].view(-1, batch['label'].shape[-1]).to(device, non_blocking=True)   # (B* seq_len, num_labels)
            mask = batch['mask'].to(device, non_blocking=True)   
            output = model(inputs, mask)
            val_loss += criterion(output, labels).item()
            probs = torch.sigmoid(output)                
            y_true.append(labels[-1,:].cpu().numpy())
            y_probs.append(probs[-1,:].cpu().numpy())
            num_batches += 1
    
    val_loss /= num_batches
   
    y_true = np.stack(y_true)
    y_probs = np.stack(y_probs)

    thresholds = find_best_thresholds(y_true, y_probs)
    y_pred = (y_probs >= thresholds).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "F1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "AUC_micro": roc_auc_score(y_true, y_pred, average='micro')
    }

    return metrics, val_loss, thresholds, num_batches


def find_best_thresholds(y_true, y_probs):
    best_thresh = []
    default_threshold = 0.5
    for i in range(y_true.shape[1]):        
        if y_true[:, i].sum() == 0:
            best_thresh.append(default_threshold)
            continue
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  

        best_idx = np.argmax(f1_scores[:-1])  
        best_thresh.append(thresholds[best_idx])
    return np.array(best_thresh)
