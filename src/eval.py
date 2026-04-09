import torch
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt 
import os
from dataset import PrefetchDataset
from loss import MaskedBCEWithLogitsLoss
from torch.utils.data import DataLoader
import time

def evaluate(model, config, test_files, criterion, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, thresholds, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    torch.set_num_threads(20)
    model.to(device)
    model.eval()
    criterion = MaskedBCEWithLogitsLoss()
    y_true, y_probs = [], []
    num_batches = 0
    test_loss = 0
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    dataset_test = PrefetchDataset(test_files, batch_size, seq_len, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, method='minmax', transform=True, shuffle=False, random_start=False, device=device)

    loader_test = DataLoader(dataset_test, batch_size=None, num_workers=0)
    with torch.no_grad():
        for batch in loader_test:
            inputs = batch['input'].to(device, non_blocking=True)   # (B, seq_len, input_dim)
            labels = batch['label'].view(-1, batch['label'].shape[-1]).to(device, non_blocking=True)   # (B* seq_len, num_labels)
            mask = batch['mask'].to(device, non_blocking=True)   
            output = model(inputs, mask)
            test_loss += criterion(output, labels).item()
            probs = torch.sigmoid(output)                
            y_true.append(labels.cpu())
            y_probs.append(probs.cpu())
            num_batches += 1
    
    test_loss /= num_batches
    
    y_true = torch.cat(y_true, dim=0)
    y_probs = torch.cat(y_probs, dim=0)
    mask_lbl = ~(y_true == -100).all(dim=1)
    y_true = y_true[mask_lbl].numpy()
    y_probs = y_probs[mask_lbl].numpy()
    y_pred = (y_probs >= thresholds).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "F1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "AUC_micro": roc_auc_score(y_true, y_pred, average='micro')
    }

    return metrics,test_loss, y_true, y_probs, y_pred, num_batches


def evaluate_sequential(model, config, test_files, criterion, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, thresholds, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    torch.set_num_threads(40)
    model.to(device)
    model.eval()
    criterion = MaskedBCEWithLogitsLoss()
    y_true, y_probs = [], []
    num_batches = 0
    test_loss = 0
    seq_len = config["seq_len"]
    batch_size = 1 
    dataset_test = PrefetchDataset(test_files, batch_size, seq_len, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, method='minmax', transform=True, shuffle=False, random_start=False, 
                              device=device, sequential=True)

    loader_test = DataLoader(dataset_test, batch_size=None, num_workers=0)
    with torch.no_grad():
        start_test = time.time()
        for batch in loader_test:
            inputs = batch['input'].to(device, non_blocking=True)   # (B, seq_len, input_dim)
            labels = batch['label'].view(-1, batch['label'].shape[-1]).to(device, non_blocking=True)   # (B* seq_len, num_labels)
            mask = batch['mask'].to(device, non_blocking=True)   
            output = model(inputs, mask)
            test_loss += criterion(output, labels).item()
            probs = torch.sigmoid(output)                
            y_true.append(labels[-1,:].cpu().numpy())
            y_probs.append(probs[-1,:].cpu().numpy())
            num_batches += 1
    
    end_test = time.time()
    test_loss /= num_batches
    print(f'Test completed with loss of {test_loss} in time {end_test-start_test}s')
   
    y_true = np.stack(y_true)
    y_probs = np.stack(y_probs)
    y_pred = (y_probs >= thresholds).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "F1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "AUC_micro": roc_auc_score(y_true, y_pred, average='micro')
    }

    return metrics,test_loss, y_true, y_probs, y_pred, num_batches

def plot_metrics_one_label(y_true, y_scores, output_dir, label_idx, signal, return_fig=False):

    y_true_label = y_true[:, label_idx]
    y_scores_label = y_scores[:, label_idx]

    precision, recall, pr_thresholds = precision_recall_curve(y_true_label, y_scores_label)
    pr_auc = auc(recall, precision)

    fpr, tpr, roc_thresholds = roc_curve(y_true_label, y_scores_label)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC plot
    axes[0].plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f})')
    axes[0].set_title('ROC curve', fontsize=20)
    axes[0].set_xlabel('False Positive Rate', fontsize=20)
    axes[0].set_ylabel('True Positive Rate', fontsize=20)
    axes[0].legend(loc='lower right', fontsize=16)
    axes[0].grid(True)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)   


    # Precision-Recall plot
    axes[1].plot(recall, precision, color='red', label=f'AUC = {pr_auc:.3f}')
    axes[1].set_title(f'Precision-Recall curve', fontsize=20)
    axes[1].set_xlabel('Recall', fontsize=20)
    axes[1].set_ylabel('Precision', fontsize=20)
    axes[1].legend(loc='lower left', fontsize=16)
    axes[1].grid(True)
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)
    
    fig.suptitle(f"Metrics with different thresholds for {signal} signal", fontsize=23)
       
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path_metrics = os.path.join(output_dir, f'metrics_{signal}.png')
        plt.savefig(fig_path_metrics, dpi=300, bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.close()

def plot_confmat_barGraph_globalIntf(y_true, y_pred, output_dir, return_fig=False):
    tp_values = []
    fp_values = []
    fn_values = []
    tn_values = []

    total_samples = y_true.shape[0]
    
    selected_labels = [6, 7, 13, 14]
    selected_signals = ['GAL_E1', 'GAL_E5a', 'GPS_L1', 'GPS_L2']
    fontsize=14 
    # Calculate TP, FP, FN, TN for each label
    for i in selected_labels:  # Iterate over each label (column)
        # True Positives (TP): where both y_true and y_pred are 1
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        tp_values.append(tp)
        
        # False Positives (FP): where y_true is 0 and y_pred is 1
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fp_values.append(fp)
        
        # False Negatives (FN): where y_true is 1 and y_pred is 0
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        fn_values.append(fn)
        
        # True Negatives (TN): where both y_true and y_pred are 0
        tn = total_samples - (tp + fp + fn)
        tn_values.append(tn)

    labels = [f'{selected_signals[i]}' for i in range(len(selected_signals))]

    x = np.arange(len(labels))  # the label positions
    width = 0.2  # width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars for TP, FP, FN
    bars1 = ax.bar(x - width * 1.5, tp_values, width, label='True Positives (TP)', color='g')
    bars2 = ax.bar(x - width / 2, fp_values, width, label='False Positives (FP)', color='gray')

    bars3 = ax.bar(x + width / 2, tn_values, width, label='True Negatives (TN)', color='b')
    bars4 = ax.bar(x + width * 1.5, fn_values, width, label='False Negatives (FN)', color='r')

    
    ax.set_xlabel(f'Interfered signals', fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    ax.set_title(f'Interfered signal detection: Per-Signal Performance (TP, FP, FN, and TN)', fontsize=23)
    ax.set_xticks(x)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(loc='best',  frameon=True, fontsize=16)
    ax.grid(True)
    
    for i, (tp, fp, fn, tn) in enumerate(zip(tp_values, fp_values, fn_values, tn_values)):        
        ax.text(x[i] - width * 1.5, tp + 0.05, str(tp), ha='center', va='bottom', color='black', fontsize=fontsize)
        ax.text(x[i] - width / 2, fp + 0.05, str(fp), ha='center', va='bottom', color='black', fontsize=fontsize)
        ax.text(x[i] + width / 2, tn + 0.05, str(tn), ha='center', va='bottom', color='black', fontsize=fontsize)
        ax.text(x[i] + width * 1.5, fn + 0.05, str(fn), ha='center', va='bottom', color='black', fontsize=fontsize)        

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path_metrics = os.path.join(output_dir, f'confMat_barGraph_InterferedSignals.png')
        plt.savefig(fig_path_metrics, dpi=300)

    if return_fig:
        return fig
    else:
        plt.close()
