# ml-based-jamming-and-spoofing-detection
## Overview

This repository supplements the publication **["Detection and Mitigation of Jamming, Meaconing, and Spoofing based on Machine Learning and Multi-Sensor Data"](https://anavs.com/support/knowledgebase/jamming-meaconing-spoofing/)** authored by:  
Philipp Bohlig, Jorge Morán García, Ramadevi Lalgudi and Jan Fischer, affiliated with the **[European Union Agency for the Space Programme (EUSPA)](https://www.euspa.europa.eu/)** and **[ANavS GmbH](https://anavs.com/)**.

It contains a dataset of pre-processed measurements collected during **[Jammertest 2024](https://jammertest.no/jammertest-2024/)**.

The dataset is hosted on Google Drive and can be downloaded here: **[data_dir](https://drive.google.com/drive/folders/1YLm2vDNVn3l4FlCIC2OuLUfzAdP0cBoz?usp=sharing)**.

It is licensed under CC BY 4.0. See LICENSE-DATA for details.

This repository provides utilities to:

- Split data into **training, validation, and test sets**
- Train on **selected subsets of files**
- Limit the number of **input features based on user input**
- Train a **CNN + LSTM neural network**
- Perform **multi-label classification**
- Evaluate **micro-averaged metrics**

---

## Dataset Description

The dataset is organized into **time-based data blocks**, where each block represents signal data collected over a specific time interval. Each folder is named using its **starting timestamp in UTC format**.

All data is stored in **`.parquet` format** for efficient storage and fast I/O performance.

### Each folder contains:

- **Input features**: pre-processed from raw GNSS receiver data(see paper for further details)
- **Label Vectors**: 48 values per input feature representing Jamming, Spoofing and Meaconing in each GNSS Signal Type(see paper for further details)

### Additional Files:

- `ToWs_in_each_file.csv`: contains the total number of input feature samples inside each file along with the start and end Gps ToWs
- `Interference_in_each_folder.csv`: contains the change of labels over non-interference and different inference period. It also has information about the total number of samples with same label and the respective time period.

---

## Code Description

It is licensed under Apache License.

Version 2.0, January 2004

http://www.apache.org/licenses/
 
Copyright (c) 2026 ANavS GmbH

### utils.py:

- **split_data():**  takes user inputs (`fraction_train, fraction_val`), splits data using information from ToWs_in_each_file.csv and writes path of train, val and test data to separate .csv files
- **get_global_features_ToWsCounts():**  reads each file to extract and combine their input features into a set, stores the number of data sample per file, allows the selection of fewer files for training and computes pos_weights from the label vectors of all/selected files

### data_to_tensors.py:

- **process_file():**  takes `global_features` (column names) as user input, extracts the columns in global_features, groups the input features w.r.t ToWs and stores input features and label vectors as tensors in .pt format
- **aggregate_stats_shapes():** aggregates min, max, mean and std from each input file and computes global min, max, mean and std values

### model.py: 

- a multilabel classifier model combining CNN and LSTM 

### loss.py: 

- BCEWithLogitsLoss() that uses pos_weights for training

### dataset.py: 

- IterableDataset() class called from PrefetchDataset() class. The IterableDataset() class inputs global values and handles padding and masking where necessary. The PrefetchDataset() class prepares data in the background, helps reduce CPU overhead and maximize GPU utilization.

### train.py:

- **train_model():** uses Adam optimizer and StepLR scheduler, trains the CNN+LSTM model using BCEWithLogitsLoss with pos_weights as criterion, writes the training loss to a tensorboard, performs validation at epoch epoch, stores the model with least validation loss and in the end test the model and evaluates performance metrics
- **plot_to_tensorboard():** creates ROC plots and stores them in tensorboard

### validate.py:

- **validate_model():** validates the trained model with BCEWithLogitsLoss criterion using dataset split into independent batches of fixed-length sequences, finds the best threshold per label and computes the performance metrics
- **validate_model_sequential():** silimar to validate_model() but validates the model using a sliding window over continuous data (overlapping sequences), preserving temporal structure. Each sequence shifts by 1 step, maintaining continuity. 
- **find_best_thresholds():** finds the best threshold rendering highest F1-score of each label

### eval.py:

- **evaluate():** evaluates the trained model with BCEWithLogitsLoss criterion using dataset split into independent batches of fixed-length sequences and computes the performance metrics
- **evaluate_sequential():** silimar to evaluate() but evaluates the model using a sliding window over continuous data (overlapping sequences), preserving temporal structure. Each sequence shifts by 1 step, maintaining continuity.
- **plot_curves():** plots ROC and Precision-Recall curves

### hyperparameter.json: 

- user defined parameters such as path of data_dir and output_dir and list of hyperparameters

### main.py:

- **load_hyperparam_combis():** reads hyperparameter.json file and expands it into all possible hyperparameter combinations
- **generate_save_globalStats():** calls fucntions in data_to_tensor.py to create .pt files with tensors, aggregate min, max, mean and std of each file with input features and stores the global values for further use
- **main():** 
- calls load_hyperparam_combis() for generating hyperparameter sweep combinations, uses check_exist flag to check for existing .parquet and .pt files of train, validation and test datasets and check_dir flag to check for existing .csv files with paths to each dataset. 
- If check_dir is false and file containing paths to train dataset is not available, calls split_data() for partitioning and saving datasets and their paths. 
- The fucntions get_global_features_ToWsCounts() and generate_save_globalStats() are called upon checking for the existence of .pt files containing global_features and the global min, max, mean and std values as well as .csv files with the file paths containing tensors of training, validation and test dataset.
- Finally the function train_model() is called with the global values and upon different configurations each containing one combination of hyperparameters  

---

## Usage:

- **To build docker image**

    `docker build -t ml_based_spoofing:latest .`

- **To create a docker container which includes all gpus and allocates atleast 16GB memory to the container**

    `docker run -it --gpus all --name spoofing_light -v /ml-based-jamming-and-spoofing-detection/:/workspace --shm-size=16g ml_based_spoofing:latest bash`

- **Inside the container**

    `cd ml-based-jamming-and-spoofing-detection`

    `CUDA_VISIBLE_DEVICES=2 python3 main.py`

- **Without docker and with or without cuda support**

    `chmod +x create_pip_venv.sh`

    `./create_pip_venv.sh`

    `source ml_env/bin/activate`

    `python3 main.py`

- **To limit the number of input features:** 

    modify the `global_features` before calling `generate_save_globalStats()`

- **To train with data from selected set of files:** 

    `selected_files = [1, 2, 5]` # The index of the files is from the list in train_files.csv, the tensors for training dataset is created for the selected set.

    `files_train = [train_files.iloc[idx, :] for idx in selected_files]`

    `files_tensor_train = [(os.path.splitext(ip_path)[0] + '_tensor.pt', os.path.splitext(lbl_path)[0] + '_tensor.pt') for ip_path, lbl_path in files_train]`

    `generate_features_indices = True`

---
    
## Note: 
- `.pt` files with tensors of input features and label vectors are used for training, validating and testing the model. They can be flexibly modified with either global_features or selected_files.