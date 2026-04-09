import torch
import torch.nn as nn

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, logits, labels):
        
        device = labels.device
        mask = (labels != self.ignore_index).float().to(device)

        cleaned_targets = torch.where(mask.bool(), labels, torch.zeros_like(labels))

        loss = self.bce(logits, cleaned_targets)  

        masked_loss = loss * mask

        valid_elements = mask.sum()

        if valid_elements > 0:
            return masked_loss.sum() / valid_elements
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)