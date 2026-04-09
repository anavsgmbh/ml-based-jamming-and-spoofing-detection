import os
import torch
import pandas as pd
from multiprocessing import Pool
import numpy as np

def process_file(in_path, global_features, generate_stats=True):
    ip_path, label_path = in_path
    ip_path_tensor = f"{os.path.splitext(ip_path)[0]}_tensor.pt"
    label_path_tensor = f"{os.path.splitext(label_path)[0]}_tensor.pt"
    
    df_ip = pd.read_parquet(ip_path, engine='pyarrow').reindex(columns=global_features)
    df_label = pd.read_parquet(label_path, columns=['TOW [s]','Label Vector'])
    
    common_ToW_values = sorted(set(df_ip['TOW [s]'].dropna().unique()) & set(df_label['TOW [s]'].dropna().unique()))
    df_ip = df_ip[df_ip['TOW [s]'].isin(common_ToW_values)]
    df_label = df_label[df_label['TOW [s]'].isin(common_ToW_values)]
    

    group_col = ['TOW [s]']
    input_cols = df_ip.columns.difference(group_col)

    df_ip_tensor = [torch.tensor(group.drop(group_col, axis=1).values, dtype=torch.float32)  
                              for _, group in df_ip.groupby(group_col)]
    label_tensor = torch.tensor(df_label['Label Vector'].to_list(), dtype=torch.float32)
    print(f'Length of ip tensor: {len(df_ip_tensor)}, Length of label tensor: {len(label_tensor)}')
    
    torch.save(df_ip_tensor, ip_path_tensor)
    torch.save(label_tensor, label_path_tensor)

    if generate_stats:        
        stats = {
            'min': df_ip[input_cols].min(skipna=True).values,
            'max': df_ip[input_cols].max(skipna=True).values,
            'mean': df_ip[input_cols].mean(skipna=True).values,
            'std': df_ip[input_cols].std(skipna=True).values,
        }
        shapes = np.array([t.shape[0:2] for t in df_ip_tensor])
        max_height, max_width = np.max(shapes, axis=0)
        global_shape = {
            'max_height': max_height,
            'max_width': max_width,
        }
        return stats, global_shape

def aggregate_stats_shapes(stats_list, global_shape):

    min_arr = np.nanmin(np.array([s['min'] for s in stats_list]), axis=0)
    max_arr = np.nanmax(np.array([s['max'] for s in stats_list]), axis=0)
    mean_arr = np.nanmean(np.array([s['mean'] for s in stats_list]), axis=0)
    std_arr = np.nanstd(np.array([s['std'] for s in stats_list]), axis=0)
    max_height = np.max(np.array([s['max_height'] for s in global_shape]), axis=0)
    max_width = np.max(np.array([s['max_width'] for s in global_shape]), axis=0)
    agg_stats = {'min': min_arr, 'max': max_arr, 'mean': mean_arr, 'std': std_arr}
    agg_shapes = {'max_height': max_height, 'max_width': max_width}
    return  agg_stats, agg_shapes

