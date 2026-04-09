import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import DataLoader
from dataset import PrefetchDataset
import os
import json
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import MultiLabelClassifierWithLstm
from loss import MaskedBCEWithLogitsLoss
from eval import * 
from validate import *
import matplotlib.pyplot as plt 
from io import BytesIO
import gc
import time

def train_model(config, train_files, val_files, test_files,  pos_weight, save_dir, target_height, target_width,
                 global_col_min, global_col_max, global_mean, global_std, transform=True, shuffle=True, random_start=True,
                 device='cuda'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using CUDA:", torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(48)

    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    patience_counter = 0
    patience = 25
    best_val_loss = float('inf')
    best_weights = None
    best_thresholds = []

    model = MultiLabelClassifierWithLstm(config['hidden_size']).to(device)
    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight.to(device), ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = StepLR(optimizer, step_size = 100, gamma=0.5, last_epoch=-1)
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
    history = {"time(s)": [],'learning_rate': [], "num_batches_train": [], "num_batches_val": [], "num_batches_test": [], "train_loss": [],  'val_loss': [], 'best_val_loss': [], 'test_loss': []} # 
    
    dataset = PrefetchDataset(train_files, batch_size, seq_len, target_height, target_width,
                    global_col_min, global_col_max, global_mean, global_std, method='minmax', transform=transform, shuffle=shuffle, random_start=random_start,
                    device=device)

    loader = DataLoader(dataset, batch_size=None, num_workers=0)  
        
    for epoch in range(config['num_epochs']):
        start_t = time.time()
        model.train()
        total_train_loss = 0.0
        time_per_epoch = 0.0
        num_batches = 0
        gc.collect()        
        for batch in loader:
            inputs = batch['input'].to(device, non_blocking=True)   # (B, seq_len, input_dim)
            labels = batch['label'].view(-1, batch['label'].shape[-1]).to(device, non_blocking=True)   # (B* seq_len, num_labels)
            mask = batch['mask'].to(device, non_blocking=True)      # (B, seq_len, input_dim)
            optimizer.zero_grad()
            out = model(inputs, mask)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
                       
            total_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches        

        val_metrics, val_loss, thresholds, num_batches_val = validate_model(model, config, val_files, criterion, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, device)
        
        scheduler.step() 
        
        lr_rate = scheduler.get_last_lr()[0]
        
        history['learning_rate'].append(lr_rate)
        end_t = time.time()
        
        time_per_epoch += (end_t-start_t)
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Number of batches: {num_batches} - Time taken: {time_per_epoch} seconds\n")
        print(f" Train Loss: {avg_train_loss: .4f} - Validation Loss: {val_loss}\n")
        
        history["train_loss"].append(avg_train_loss)
        history["time(s)"].append(time_per_epoch) 
        history["num_batches_train"].append(num_batches) 

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        history['val_loss'].append(val_loss)
        history['num_batches_val'].append(num_batches_val)

        writer.add_scalar("Loss/Validation", val_loss, epoch)

        writer.add_scalar('Learning Rate', lr_rate, epoch)
        writer.add_scalar('Number of training batches', num_batches, epoch)

        
        for k, v in val_metrics.items():
            writer.add_scalar(f"Validation/{k}", v, epoch) 

        if epoch > 150:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())  
                best_thresholds = thresholds
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}, Best_val_loss: {best_val_loss} at epoch {best_epoch}")
            break
    
    torch.save({
        'model_state_dict': best_weights,
        'thresholds': best_thresholds 
    }, os.path.join(save_dir,'best_model_with_thresholds.pth'))

    history['best_val_loss'].append(best_val_loss)     
    
    model.load_state_dict(best_weights)

    val_metrics, val_loss, thresholds, num_batches_val = validate_model(model, config, val_files, criterion, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, device)

    with open(os.path.join(save_dir, "val.metrics.json"), 'w') as f:
        json.dump(val_metrics, f, indent=2) 

    test_metrics, test_loss, y_true, y_scores, y_pred, num_batches_test = evaluate_sequential(model, config, test_files, criterion, target_height, target_width, global_col_min, global_col_max, 
                              global_mean, global_std, best_thresholds, device=device)
    
    history['test_loss'].append(test_loss)
    history['num_batches_test'].append(num_batches_test)

    with open(os.path.join(save_dir, "test.metrics.json"), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    with open(os.path.join(save_dir, "loss_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    for k, v in test_metrics.items():
        writer.add_scalar(f"Test/{k}", v, 0)

    
    
    def plot_to_tensorboard(fig):
        from PIL import Image
        buf = BytesIO()
        fig.savefig(buf, format= 'png', bbox_inches= 'tight')
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2,0,1).contiguous()
        image_tensor = image_tensor.to(dtype=torch.uint8)
        plt.close(fig)
        return image_tensor
    
    y_true_grouped = y_true.reshape(-1,16,3)
    y_true_grouped = (np.max(y_true_grouped,axis=2)).astype(int)

    y_scores_grouped = y_scores.reshape(-1,16,3)
    y_scores_grouped = (np.max(y_scores_grouped,axis=2))

    y_pred_grouped = y_pred.reshape(-1,16,3)
    y_pred_grouped = (np.max(y_pred_grouped,axis=2)).astype(int)

    # Plot metrics for GPS_L2   
    plt_fpr_tpr_L2 = plot_metrics_one_label(y_true_grouped, y_scores_grouped, save_dir, 14, 'GPS_L2', return_fig=True)
    image_tensor_L2 = plot_to_tensorboard(plt_fpr_tpr_L2)
    writer.add_image("Evaluation/ROC_PR for GPS_L2", image_tensor_L2, global_step=0)

    # Plot Confusion Matrix in bar graph
    plt_confMat = plot_confmat_barGraph_globalIntf(y_true_grouped, y_pred_grouped, save_dir, return_fig=True)
    image_confMat = plot_to_tensorboard(plt_confMat)
    writer.add_image("Interfered signal detection: Per-Signal Performance (TP, FP, FN, and TN)", image_confMat, global_step=0)

