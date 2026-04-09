import os
import pandas as pd 
import numpy as np 
import itertools
import torch

def append_file_path(data_dir, dataset_type, check_exist, *args):
    if dataset_type == 'train':
        train_files = []        
        unique_subfolders = args[0]
        subfolder_train_val = args[1]
        till_ToW_to_train = args[2]
        
        for sub in unique_subfolders:    
            label_file_path = os.path.join(data_dir,sub, f'df_labelVectors.parquet')
            ip_file_path = os.path.join(data_dir,sub, f'df_inputFeatures.parquet')
            train_files.append((ip_file_path,label_file_path))

        ip_file_path_train = os.path.join(data_dir,subfolder_train_val, f'df_inputFeatures_train.parquet')
        label_file_path_train = os.path.join(data_dir,subfolder_train_val, f'df_labelVectors_train.parquet')
        
        train_files.append((ip_file_path_train, label_file_path_train))
        
        if (os.path.exists(ip_file_path_train) & check_exist):
            return train_files
        else:
            split_file_add_path(os.path.join(data_dir,subfolder_train_val, f'df_inputFeatures.parquet'), 
                            os.path.join(data_dir,subfolder_train_val, f'df_labelVectors.parquet'), 'train_val',
                            till_ToW_to_train)
            return train_files 
        
    elif dataset_type == 'val':
        val_files = []
        unique_subfolders = args[0]
        subfolder_train_val = args[1]
        subfolder_val_test = args[2]
        till_ToW_to_val = args[3]
        
        ip_file_path_val = os.path.join(data_dir,subfolder_train_val, f'df_inputFeatures_val.parquet')
        label_file_path_val = os.path.join(data_dir,subfolder_train_val, f'df_labelVectors_val.parquet')
        val_files.append((ip_file_path_val, label_file_path_val))
        
        for sub in unique_subfolders:    
            label_file_path = os.path.join(data_dir,sub, f'df_labelVectors.parquet')
            ip_file_path = os.path.join(data_dir,sub, f'df_inputFeatures.parquet')
            val_files.append((ip_file_path,label_file_path))

        if (subfolder_train_val != subfolder_val_test):
            ip_file_path_val = os.path.join(data_dir,subfolder_val_test, f'df_inputFeatures_val.parquet')
            label_file_path_val = os.path.join(data_dir,subfolder_val_test, f'df_labelVectors_val.parquet')
            val_files.append((ip_file_path_val, label_file_path_val))
        
            if (os.path.exists(ip_file_path_val) & check_exist):
                return val_files
        
        split_file_add_path(os.path.join(data_dir,subfolder_val_test, f'df_inputFeatures.parquet'), 
                            os.path.join(data_dir,subfolder_val_test, f'df_labelVectors.parquet'), 'val_test',
                            till_ToW_to_val)
        return val_files
    else:
        test_files = []
        unique_subfolders = args[0]
        subfolder_val_test = args[1]
        
        ip_file_path_test = os.path.join(data_dir,subfolder_val_test, f'df_inputFeatures_test.parquet')
        label_file_path_test = os.path.join(data_dir,subfolder_val_test, f'df_labelVectors_test.parquet')
        test_files.append((ip_file_path_test, label_file_path_test))
        
        for sub in unique_subfolders:    
            label_file_path = os.path.join(data_dir,sub, f'df_labelVectors.parquet')
            ip_file_path = os.path.join(data_dir,sub, f'df_inputFeatures.parquet')
            test_files.append((ip_file_path,label_file_path))
        return test_files

def split_file_add_path(ip_path, label_path, split_type, till_ToW):
    df_ip = pd.read_parquet(ip_path)
    df_label = pd.read_parquet(label_path)
    till_ToW = till_ToW + 0.8
    if split_type == 'train_val':        
        df_ip_train = df_ip[df_ip['TOW [s]'] <= till_ToW]
        df_ip_val = df_ip[df_ip['TOW [s]'] > till_ToW]

        df_label_train = df_label[df_label['TOW [s]'] <= till_ToW]
        df_label_val = df_label[df_label['TOW [s]'] > till_ToW]

        df_ip_train.to_parquet(os.path.splitext(ip_path)[0]+ f'_train.parquet', index=False)
        df_ip_val.to_parquet(os.path.splitext(ip_path)[0]+ f'_val.parquet', index=False)
        unique_ToWs = df_ip_train['TOW [s]'].drop_duplicates().sort_values().reset_index(drop=True)
        print("Unique ToWs in train file", len(unique_ToWs))

        df_label_train.to_parquet(os.path.splitext(label_path)[0]+ f'_train.parquet', index=False)
        df_label_val.to_parquet(os.path.splitext(label_path)[0]+ f'_val.parquet', index=False)
    else:
        df_ip_test = df_ip[df_ip['TOW [s]'] > till_ToW]
        df_label_test = df_label[df_label['TOW [s]'] > till_ToW]
        unique_ToWs = df_ip_test['TOW [s]'].drop_duplicates().sort_values().reset_index(drop=True)
        print("Unique ToWs in test file", len(unique_ToWs))
        df_ip_test.to_parquet(os.path.splitext(ip_path)[0]+ f'_test.parquet', index=False)
        df_label_test.to_parquet(os.path.splitext(label_path)[0]+ f'_test.parquet', index=False)

        ip_file_path_val = os.path.splitext(ip_path)[0]+ f'_val.parquet'
        label_file_path_val = os.path.splitext(label_path)[0]+ f'_val.parquet'
        if not (os.path.exists(ip_file_path_val) or os.path.exists(label_file_path_val)):
            df_ip_val = df_ip[df_ip['TOW [s]'] <= till_ToW]
            df_label_val = df_label[df_label['TOW [s]'] <= till_ToW]
            unique_ToWs = df_ip_val['TOW [s]'].drop_duplicates().sort_values().reset_index(drop=True)
            print("Unique ToWs in val file", len(unique_ToWs))
            df_ip_val.to_parquet(ip_file_path_val, index=False)        
            df_label_val.to_parquet(label_file_path_val, index=False)
        else:
            df_ip_val = pd.read_parquet(ip_file_path_val)
            df_label_val = pd.read_parquet(label_file_path_val)
            df_ip_val = df_ip_val[df_ip_val['TOW [s]'] <= till_ToW]
            df_label_val = df_label_val[df_label_val['TOW [s]'] <= till_ToW]
            unique_ToWs = df_ip_val['TOW [s]'].drop_duplicates().sort_values().reset_index(drop=True)
            print("Unique ToWs in val file", len(unique_ToWs))
            df_ip_val.to_parquet(ip_file_path_val, index=False)        
            df_label_val.to_parquet(label_file_path_val, index=False)
        

def split_data(data_dir, check_exist, fraction_train, fraction_val):
    df_ToWs_in_file = pd.read_csv(os.path.join(data_dir, f'ToWs_in_each_file.csv'), index_col=False) 
    total_ToWs_ip = sum(df_ToWs_in_file['len_Unique_Tows'])
    train_files = []
    val_files = []
    test_files = []
    total_train = 0
    num_train = round(total_ToWs_ip * fraction_train) 
    num_val = round(total_ToWs_ip * fraction_val) 
    i = 0
    while total_train < num_train:
        total_train += df_ToWs_in_file['len_Unique_Tows'][i] 
        i += 1
    row_to_split_train_val = i-1    
    unique_subfolders_train = df_ToWs_in_file['Sub_folder'][:row_to_split_train_val].unique()
    subfolder_split_train_val = df_ToWs_in_file['Sub_folder'][row_to_split_train_val] 
    if subfolder_split_train_val in unique_subfolders_train:
        unique_subfolders_train = unique_subfolders_train[unique_subfolders_train != subfolder_split_train_val]    
        
    total_val = total_train - num_train 
    till_ToW_to_train = int(df_ToWs_in_file['TOW [s]_end'][row_to_split_train_val]) - total_val 
    
    while total_val < num_val:
        total_val += df_ToWs_in_file['len_Unique_Tows'][i] 
        i += 1
   
    row_to_split_val_test = i-1
    unique_subfolders_val = df_ToWs_in_file['Sub_folder'][row_to_split_train_val+1:row_to_split_val_test+1].unique()
    subfolder_split_val_test = df_ToWs_in_file['Sub_folder'][row_to_split_val_test]
    if subfolder_split_val_test in unique_subfolders_val:
        unique_subfolders_val = unique_subfolders_val[unique_subfolders_val != subfolder_split_val_test]
    if subfolder_split_train_val in unique_subfolders_val: 
        unique_subfolders_val = unique_subfolders_val[unique_subfolders_val != subfolder_split_train_val]
    total_test = total_val - num_val
    till_ToW_to_val = int(df_ToWs_in_file['TOW [s]_end'][row_to_split_val_test]) - total_test
    row_test_set = i
    
    total_test += sum(df_ToWs_in_file['len_Unique_Tows'][row_test_set::])
    unique_subfolders_test = df_ToWs_in_file['Sub_folder'][row_test_set:].unique()
    if subfolder_split_val_test in unique_subfolders_test:
        unique_subfolders_test = unique_subfolders_test[unique_subfolders_test != subfolder_split_val_test]
    

    train_files = append_file_path(data_dir, 'train', check_exist, unique_subfolders_train, subfolder_split_train_val, till_ToW_to_train)
    val_files = append_file_path(data_dir, 'val', check_exist, unique_subfolders_val, subfolder_split_train_val, subfolder_split_val_test, till_ToW_to_val)
    test_files = append_file_path(data_dir, 'test', check_exist, unique_subfolders_test, subfolder_split_val_test)
    train_files = pd.DataFrame(train_files)
    val_files = pd.DataFrame(val_files)
    test_files = pd.DataFrame(test_files)
    train_files.to_csv(os.path.join(data_dir, 'train_files.csv'), index=False)
    val_files.to_csv(os.path.join(data_dir, 'val_files.csv'), index=False)
    test_files.to_csv(os.path.join(data_dir, 'test_files.csv'), index=False)
    return train_files, val_files, test_files

def get_global_features_ToWsCounts(data_dir, file_paths, num_labels, selected_files):
    feature_set = set()
    global_indices_len = []
    pos_counts = torch.zeros(num_labels)
    neg_counts = torch.zeros(num_labels)
    for idx in selected_files:
        input_file_path, label_file_path = file_paths.iloc[idx,:]
        df_ip = pd.read_parquet(input_file_path, engine='pyarrow')
        feature_set.update(df_ip.columns)
        unique_ToWs = df_ip['TOW [s]'].drop_duplicates().sort_values().reset_index(drop=True)
        len_unique_ToWs = len(unique_ToWs)
        global_indices_len.append(len_unique_ToWs)
        df_labels = pd.read_parquet(label_file_path, engine='pyarrow')
        label_tensor = torch.tensor(df_labels['Label Vector'].tolist())
        pos_counts += label_tensor.sum(dim=0)
        neg_counts += label_tensor.size(0) - label_tensor.sum(dim=0)                
    to_remove = {'WNc [w]', 'UTC_Time'}
    feature_set -= to_remove
    mask_pos = pos_counts > 0.0
    pos_weight = torch.ones_like(pos_counts) + 3.0
    pos_weight[mask_pos] = torch.log1p(neg_counts[mask_pos] / (pos_counts[mask_pos]))
    torch.save({'global_features': sorted(feature_set), 'global_indices_len': global_indices_len}, os.path.join(data_dir, f'global_features_lenIndices.pt'))
    torch.save(pos_weight, os.path.join(data_dir, f'pos_weight.pt'))
    
    return sorted(feature_set), global_indices_len, pos_weight
