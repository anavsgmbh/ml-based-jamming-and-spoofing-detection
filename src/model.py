import torch
import torch.nn as nn

class MultiLabelClassifierWithLstm(nn.Module):
    def __init__(self, hidden_size, num_layers=1, kernel_size=7, padding=3, stride=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=kernel_size, padding=padding, stride=stride), # Num_params = 16 + 16*7*7*2 = 1584   
            nn.BatchNorm2d(16), # Num_params = 16+16 = 32        
            nn.ReLU(), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Num_params = 32 + 32*3*3*16 = 4640
            nn.BatchNorm2d(32), # Num_params = 32+32 = 64
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Num_params = 64 + 64*3*3*32 = 18496
            nn.BatchNorm2d(64), # Num_params = 64+64 = 128
            nn.ReLU()          
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True) # First layer, Num_params = 4*(hidden_size^2 + 65*hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)  # Num_params = 2*hidden_size
        
        self.fc = nn.Linear(hidden_size, 48) # Num_params = 48 + hidden_size*48 
    
    def forward(self, ip_features, mask_tensors):
        batch_size = ip_features.shape[0]
        seq_len = ip_features.shape[1]

        input_features = ip_features.reshape(-1, 1, ip_features.shape[-2], ip_features.shape[-1])
        mask = mask_tensors.reshape(-1, 1, mask_tensors.shape[-2], mask_tensors.shape[-1])

        input_features_mask = torch.cat([input_features, mask], dim=1)
        
        out = self.cnn(input_features_mask.float()) # Total params = 1584 + 32 + 4640 + 64 + 18496 + 128 = 24944
       
        # Manually masked Average Pooling to accomodate missing values
        mask_float = mask.float()
        avg_pooled = (out * mask_float).sum(dim= [2,3]) / mask_float.sum(dim=[2,3]).clamp(min=1e-6) # Shape [B*N, 64]
        out = avg_pooled.view(batch_size, seq_len, -1) # Shape [B,N,64]
        out, _ = self.lstm(out) # Shape [B, N, hidden_size]
        out = self.layer_norm(out) 
        in_fc = out.reshape(-1, out.shape[-1]) # Shape [B.N, hidden_size]
        out = self.fc(in_fc) # Shape [B.N, 48]
        return out.squeeze(-1) if out.shape[-1] == 1 else out
