import os
import torch
from torch.utils.data import IterableDataset
import random
import threading
import queue

def pad_to_shape(tensor, target_shape, padding_value=0):
    
    cur_h, cur_w = tensor.shape
    tgt_h, tgt_w = target_shape

    pad_h = tgt_h - cur_h
    pad_w = tgt_w - cur_w

    padding = (0, int(pad_w), 0, int(pad_h))
    return torch.nn.functional.pad(tensor, padding, value=padding_value)

class SeqBatchDataset(IterableDataset):
    def __init__(self, files, batch_size, seq_len, target_height, target_width,
                 global_col_min, global_col_max, global_mean, global_std, method='minmax', 
                 transform=True, shuffle=True, random_start=True,device='cuda', sequential=False):
        
        self.files = files
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.target_height = target_height
        self.target_width = target_width
        self.global_col_min = global_col_min.to(device)
        self.global_col_max = global_col_max.to(device)
        self.global_mean = global_mean.to(device)
        self.global_std = global_std.to(device)
        self.transform = transform
        self.shuffle = shuffle
        self.random_start = random_start
        self.device = device
        self.method = method
        self.sequential = sequential

    def normalize_inputs(self, inputs, feature_range=(0.001, 1)):
        lengths = [x.size(0) for x in inputs]

        inputs = torch.cat(inputs, dim=0).to(self.device)
        
        if not self.transform:
            mask = ~torch.isnan(inputs)
            inputs = list(torch.split(inputs, lengths, dim=0))
            mask = list(torch.split(mask.float(), lengths, dim=0))
            return inputs, mask 
        
        mask = ~torch.isnan(inputs)
        normalized = torch.zeros_like(inputs, device=self.device)
        if self.method == 'zscore':
            std = self.global_std.clone()
            std[std == 0] = 1
            normalized[mask] = ((inputs - self.global_mean.view(1, -1)) / self.global_std.view(1, -1))[mask]

        elif self.method == 'minmax':
            a, b = feature_range
            scale = self.global_col_max - self.global_col_min
            scale[scale == 0] = 1 

            normalized = a + ((inputs - self.global_col_min.view(1, -1)) / scale.view(1, -1)) * (b - a)
            normalized[~mask] = 0.0
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        normalized = list(torch.split(normalized, lengths, dim=0))
        mask = list(torch.split(mask.float(), lengths, dim=0))

        return normalized, mask 

    def __iter__(self):
        file_list = self.files.copy()
        if self.shuffle:
            random.shuffle(file_list)

        batch_inputs, batch_labels, batch_masks = [], [], []

        for input_path, label_path in file_list:
            inputs = torch.load(input_path)
            labels = torch.load(label_path)

            inputs, mask = self.normalize_inputs(inputs)

            length = len(inputs)

            padded_inputs = []
            padded_masks = []
            for i in range(length):
                padded_input = pad_to_shape(inputs[i], (self.target_height, self.target_width), padding_value=0)
                padded_mask = pad_to_shape(mask[i], (self.target_height, self.target_width), padding_value=0)
                padded_inputs.append(padded_input)
                padded_masks.append(padded_mask)
            inputs = torch.stack(padded_inputs, dim=0) 
            masks = torch.stack(padded_masks, dim=0)
            if length < self.seq_len:
                inputs = torch.nn.functional.pad(inputs, (0, 0, 0, 0, 0, self.seq_len - length), value=0) 
                masks = torch.nn.functional.pad(masks, (0, 0, 0, 0, 0, self.seq_len - length), value=0)
                pad_label_shape = list(labels.shape)
                pad_label_shape[0] = self.seq_len - length
                pad_label = torch.full(pad_label_shape, fill_value=-100, dtype=labels.dtype)
                labels = torch.cat([labels, pad_label], dim=0)
                sequences = [(inputs, masks, labels)]
            else:
                sequences = []
                if self.random_start:
                    start_idx = random.randint(0, length - self.seq_len)
                else:
                    start_idx = 0

                idx = start_idx
                if self.sequential:
                    remainder = self.seq_len - (idx + 1) 
                    while remainder >= 0:
                        idx += 1
                        seq_input = torch.nn.functional.pad(inputs[:idx], (0, 0, 0, 0, remainder, 0), value=0)
                        seq_masks = torch.nn.functional.pad(masks[:idx], (0, 0, 0, 0, remainder, 0), value=0)
                        pad_label_shape = list(labels[:idx].shape)
                        pad_label_shape[0] = remainder
                        pad_label = torch.full(pad_label_shape, fill_value=-100, dtype=labels.dtype)
                        seq_label = torch.cat([pad_label, labels[:idx] ], dim=0)
                        sequences.append((seq_input, seq_masks, seq_label))
                        remainder = self.seq_len - (idx + 1)
                    while remainder < 0 and idx+1 <= length:
                        idx += 1
                        seq_input = inputs[idx-self.seq_len:idx]
                        seq_masks = masks[idx-self.seq_len:idx]
                        seq_label = labels[idx-self.seq_len:idx]
                        sequences.append((seq_input, seq_masks, seq_label))                        
                        remainder = self.seq_len - (idx + 1)
                                        
                else:
                    while idx + self.seq_len <= length:
                        seq_input = inputs[idx:idx+self.seq_len]
                        seq_masks = masks[idx:idx+self.seq_len]
                        seq_label = labels[idx:idx+self.seq_len]
                        sequences.append((seq_input, seq_masks, seq_label))
                        idx += self.seq_len

                    remainder = length - idx
                    if remainder > 0:
                        seq_input = torch.nn.functional.pad(inputs[idx:], (0, 0, 0, 0, 0, self.seq_len - remainder), value=0)
                        seq_masks = torch.nn.functional.pad(masks[idx:], (0, 0, 0, 0, 0, self.seq_len - remainder), value=0)
                        pad_label_shape = list(labels[idx:].shape)
                        pad_label_shape[0] = self.seq_len - remainder
                        pad_label = torch.full(pad_label_shape, fill_value=-100, dtype=labels.dtype)
                        seq_label = torch.cat([labels[idx:], pad_label], dim=0)
                        sequences.append((seq_input, seq_masks, seq_label))
                
                if self.shuffle:
                    random.shuffle(sequences)    

            for input_seq, mask_seq, label_seq in sequences:
                batch_inputs.append(input_seq)
                batch_labels.append(label_seq)
                batch_masks.append(mask_seq)

                if len(batch_inputs) == self.batch_size:
                    yield {
                        'input': torch.stack(batch_inputs).to(self.device),   # (batch_size, seq_len, H, W)
                        'label': torch.stack(batch_labels).to(self.device),
                        'mask': torch.stack(batch_masks).to(self.device)
                    }
                    batch_inputs, batch_labels, batch_masks = [], [], []

        if batch_inputs:
            yield {
                'input': torch.stack(batch_inputs).to(self.device),
                'label': torch.stack(batch_labels).to(self.device),
                'mask': torch.stack(batch_masks).to(self.device)
            }

class PrefetchDataset(SeqBatchDataset):
    def __init__(self, *args, prefetch_size=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()

    def _producer(self):
        try:
            for batch in super().__iter__():
                self.queue.put(batch)
                if self.stop_event.is_set():
                    break
        finally:
            self.queue.put(None)

    def __iter__(self):
        self.stop_event.clear()
        thread = threading.Thread(target=self._producer)
        thread.daemon = True
        thread.start()

        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch

        self.stop_event.set()
        thread.join()
