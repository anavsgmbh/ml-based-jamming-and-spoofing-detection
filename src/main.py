import os
import json
from utils import *
import itertools
from datetime import datetime
from train import train_model
import torch
from multiprocessing import Pool
from functools import partial
from data_to_tensors import *

def load_hyperparam_combis(path):
    with open(path) as f: 
        params = json.load(f)
    
    sweep_params = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in params.items() if not isinstance(v, list)}

    if not sweep_params:
        return [params]

    keys, values = zip(*sweep_params.items())
    combis = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_configs = []
    for combi in combis:
        config = fixed_params.copy()
        config.update(combi)
        all_configs.append(config)
    
    return all_configs

def generate_save_globalStats(files, global_features, generate_stats, num_workers, global_stats_file_path=None, global_shape_file_path=None):    
    process_partial = partial(process_file, global_features=global_features, generate_stats=generate_stats)
    with Pool(num_workers) as pool:
        results = pool.map(process_partial, files)
    
    if generate_stats:            
        stats_list, global_shape = zip(*results)

        agg_stats, agg_shapes = aggregate_stats_shapes(stats_list, global_shape)

        torch.save({
            'min': torch.tensor(agg_stats['min'], dtype=torch.float32),
            'max': torch.tensor(agg_stats['max'], dtype=torch.float32),
            'mean': torch.tensor(agg_stats['mean'], dtype=torch.float32),
            'std': torch.tensor(agg_stats['std'], dtype=torch.float32),
        }, global_stats_file_path)

        torch.save({
            'max_height': torch.tensor(agg_shapes['max_height'], dtype=torch.float32),
            'max_width': torch.tensor(agg_shapes['max_width'], dtype=torch.float32),
        }, global_shape_file_path) 
        return agg_stats, agg_shapes

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    configs = load_hyperparam_combis(os.path.join(current_dir, f"hyperparams.json"))
    data_dirs = os.path.join(parent_dir, configs[0]['data_dir'])
    check_exist = True
    check_dir = True
    if os.path.exists(os.path.join(data_dirs, 'train_files.csv')) and check_dir:
        train_files = pd.read_csv(os.path.join(data_dirs, 'train_files.csv'), index_col=False)
        val_files = pd.read_csv(os.path.join(data_dirs, 'val_files.csv'), index_col=False)
        test_files = pd.read_csv(os.path.join(data_dirs, 'test_files.csv'), index_col=False)
    else:
        train_files, val_files, test_files = split_data(data_dirs, check_exist, fraction_train=0.8, fraction_val=0.10)
    
    global_features = None
    features_indices_file_path = os.path.join(data_dirs, f'global_features_lenIndices.pt')
    global_stats_file_path = os.path.join(data_dirs, f'normalization_stats.pt')
    global_shape_file_path = os.path.join(data_dirs, f'global_shape.pt')
    num_workers = 8
    selected_files = np.arange(len(train_files)) # [1, 2, 5]
    files_train = [train_files.iloc[idx, :] for idx in selected_files]
    files_tensor_train = [(os.path.splitext(ip_path)[0] + '_tensor.pt', os.path.splitext(lbl_path)[0] + '_tensor.pt') for ip_path, lbl_path in files_train]

    generate_features_indices = False
    generate_stats_shape = False 
    generate_stats = True

    if os.path.exists(features_indices_file_path):
        features_indices = torch.load(features_indices_file_path)
        global_features = features_indices['global_features']
        global_indices_len = features_indices['global_indices_len']
        if (len(selected_files) == len(global_indices_len)):            
            print(f'File with Global features and length of indices exists \n')
            print(f'Global features: {global_features}\n')
            for i in range(len(global_indices_len)):
                print(f'Number of unique ToWs in each training sequence: {global_indices_len[i]}\n')

            print(f'Total number of unique ToWs in training sequences: {sum(global_indices_len)}\n')
            pos_weight = torch.load(os.path.join(data_dirs, f'pos_weight.pt'))
            print(f'pos_weights:{pos_weight}')

            if(os.path.exists(global_stats_file_path)):
                print(f'File with global stats and shape exists \n')
                agg_stats = torch.load(global_stats_file_path)
                global_shape = torch.load(global_shape_file_path)
                target_height = global_shape['max_height']
                target_width = global_shape['max_width']
                min_col_arr = agg_stats['min']
                max_col_arr = agg_stats['max']
                global_mean = agg_stats['mean']
                global_std = agg_stats['std']
                print(f'Height and width of each tensor: {target_height}, {target_width}')
                print(f'Global minimum values: {min_col_arr}')
                print(f'Global maximum values: {max_col_arr}')
            else:
                generate_stats_shape = True
        else:
                generate_features_indices = True
    else:
        generate_features_indices=True

    if generate_features_indices:
        num_labels = 48
        global_features, global_indices_len, pos_weight = get_global_features_ToWsCounts(data_dirs, train_files, num_labels, selected_files)
          
        agg_stats, global_shape = generate_save_globalStats(files_train, global_features, generate_stats, num_workers, global_stats_file_path, global_shape_file_path)

        target_height = torch.tensor(global_shape['max_height'], dtype=torch.float32)
        target_width = torch.tensor(global_shape['max_width'], dtype=torch.float32)
        min_col_arr = torch.tensor(agg_stats['min'], dtype=torch.float32)
        max_col_arr = torch.tensor(agg_stats['max'], dtype=torch.float32)
        global_mean = torch.tensor(agg_stats['mean'], dtype=torch.float32)
        global_std = torch.tensor(agg_stats['std'], dtype=torch.float32) 

        print(f'Global features: {global_features}\n')
        for i in range(len(global_indices_len)):
            print(f'Number of unique ToWs in each training sequence: {global_indices_len[i]}\n')

        print(f'Total number of unique ToWs in training sequences: {sum(global_indices_len)}\n')
        
        print(f'pos_weights:{pos_weight}')

        print(f'Height and width of each tensor: {target_height}, {target_width}')
        print(f'Global minimum values: {min_col_arr}')
        print(f'Global maximum values: {max_col_arr}')

    # To select limited set of features eg: without IMU measurements
    # global_features = ['CN0_dBHz [dB-Hz]_0', 'CN0_dBHz [dB-Hz]_1', 'CN0_dBHz [dB-Hz]_10', 'CN0_dBHz [dB-Hz]_11', 'CN0_dBHz [dB-Hz]_12', 'CN0_dBHz [dB-Hz]_13', 'CN0_dBHz [dB-Hz]_14', 'CN0_dBHz [dB-Hz]_15', 'CN0_dBHz [dB-Hz]_16', 'CN0_dBHz [dB-Hz]_17', 'CN0_dBHz [dB-Hz]_18', 'CN0_dBHz [dB-Hz]_2', 'CN0_dBHz [dB-Hz]_3', 'CN0_dBHz [dB-Hz]_4', 'CN0_dBHz [dB-Hz]_5', 'CN0_dBHz [dB-Hz]_6', 'CN0_dBHz [dB-Hz]_7', 'CN0_dBHz [dB-Hz]_8', 'CN0_dBHz [dB-Hz]_9', 'CR_values_0', 'CR_values_1', 'CR_values_10', 'CR_values_11', 'CR_values_12', 'CR_values_13', 'CR_values_14', 'CR_values_15', 'CR_values_16', 'CR_values_17', 'CR_values_18', 'CR_values_2', 'CR_values_3', 'CR_values_4', 'CR_values_5', 'CR_values_6', 'CR_values_7', 'CR_values_8', 'CR_values_9', 'Constellation', 'Constellation_Freq', 'Gain [dB]', 'Prn_0', 'Prn_1', 'Prn_10', 'Prn_11', 'Prn_12', 'Prn_13', 'Prn_14', 'Prn_15', 'Prn_16', 'Prn_17', 'Prn_18', 'Prn_2', 'Prn_3', 'Prn_4', 'Prn_5', 'Prn_6', 'Prn_7', 'Prn_8', 'Prn_9', 'SampleVar', 'TOW [s]']
    # print("Running with limited set of features: ", global_features)
    # generate_stats_shape = True
    # check_exist = False

    if generate_stats_shape: 
        agg_stats, global_shape = generate_save_globalStats(files_train, global_features, generate_stats, num_workers, global_stats_file_path, global_shape_file_path)

        target_height = torch.tensor(global_shape['max_height'], dtype=torch.float32)
        target_width = torch.tensor(global_shape['max_width'], dtype=torch.float32)
        min_col_arr = torch.tensor(agg_stats['min'], dtype=torch.float32)
        max_col_arr = torch.tensor(agg_stats['max'], dtype=torch.float32)
        global_mean = torch.tensor(agg_stats['mean'], dtype=torch.float32)
        global_std = torch.tensor(agg_stats['std'], dtype=torch.float32)
        print(f'Height and width of each tensor: {target_height}, {target_width}')
        print(f'Global minimum values: {min_col_arr}')
        print(f'Global maximum values: {max_col_arr}')     


    files_val = val_files.to_numpy().tolist()    
    files_tensor_val = [(os.path.splitext(ip_path)[0] + '_tensor.pt', os.path.splitext(lbl_path)[0] + '_tensor.pt') for ip_path, lbl_path in files_val]
    files_test = test_files.to_numpy().tolist() 
    files_tensor_test = [(os.path.splitext(ip_path)[0] + '_tensor.pt', os.path.splitext(lbl_path)[0] + '_tensor.pt') for ip_path, lbl_path in files_test]
    train_tensor_files_exist = all(os.path.exists(p) for tup in files_tensor_train for p in tup)
    val_tensor_files_exist = all(os.path.exists(p) for tup in files_tensor_val for p in tup)
    test_tensor_files_exist = all(os.path.exists(p) for tup in files_tensor_test for p in tup)
    
    if train_tensor_files_exist:
        print('Training tensor files exists \n')
    else:
        print('Some of the training tensor files are missing \n')
        
    if val_tensor_files_exist and check_exist:
        print('Validation tensor files exists \n')
    else:
        generate_stats = False
        generate_save_globalStats(files_val, global_features, generate_stats, num_workers)

    if test_tensor_files_exist and check_exist:
        print('Test tensor files exists \n')
    else:
        generate_stats = False
        generate_save_globalStats(files_test, global_features, generate_stats, num_workers)

    for i, config in enumerate(configs):
        print(f"\n Running config {i+1}/{len(configs)}:")
        print(json.dumps(config, indent=2))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param1 = config["seq_len"]
        param2 = config["hidden_size"]
        config_id = f"run_{i+1}_{timestamp}_{param1}_{param2}" 
        config["save_dir"] = os.path.join(parent_dir, config.get("save_dir_base", "./output"), config_id)

        train_model(config, files_tensor_train, files_tensor_val, files_tensor_test,  pos_weight, config["save_dir"], target_height, target_width,
                 min_col_arr, max_col_arr, global_mean, global_std)

if __name__ == "__main__":
    main()
