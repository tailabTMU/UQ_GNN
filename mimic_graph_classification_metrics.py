import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tabulate import tabulate
from utils.gnn import mc_inference_heterogeneous as mc_inference
import torch
from datasets import MIMICDataset
from torch_geometric.loader import DataLoader
from models.mimic_models import GraphConvGNN, SelfDistillationGraphConvGNN, MCDropoutGraphConvGNN
from sklearn.model_selection import StratifiedKFold
import random
import time
from collections import Counter
from sklearn.utils import resample


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Function to compute majority vote for each sample
def majority_vote(predictions):
    # Unique values and their counts
    unique, counts = np.unique(predictions, return_counts=True)
    # Return the value with the highest count
    return unique[np.argmax(counts)]


def smooth_probabilities(predictions, epsilon=1e-8):
    # Add epsilon to all probabilities
    smoothed = predictions + epsilon
    # Renormalize along the class dimension (last dimension)
    smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True)
    return smoothed


def smooth_probabilities_np(predictions, epsilon=1e-8):
    # Add epsilon to all probabilities
    smoothed = predictions + epsilon
    # Renormalize along the class dimension (last dimension)
    smoothed = smoothed / smoothed.sum(axis=-1, keepdims=True)
    return smoothed


def compute_calibration_metrics(predictions, true_labels):
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    # Ensure binary predictions shape is [n_samples]
    if predictions.ndim == 2:
        predictions = predictions[:, 1]  # Take probability of positive class

    # Clip predictions to avoid log(0) issues
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

    # Binary accuracy calculation
    accuracy = ((predictions >= 0.5) == true_labels).astype(float)

    # Confidence calculation for binary classification
    confidence = np.maximum(predictions, 1 - predictions)

    bins = np.linspace(0, 1, 11)  # 10 bins from 0.0 to 1.0
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_acc = []
    bin_conf = []
    bin_count = []

    for i in range(len(bins) - 1):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc.append(accuracy[mask].mean())
            bin_conf.append(confidence[mask].mean())
            bin_count.append(np.sum(mask))
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_count.append(0)

    valid = ~np.isnan(bin_acc)

    # Expected Calibration Error (ECE)
    ece = np.sum(
        (np.array(bin_count)[valid] / np.sum(bin_count)) *
        np.abs(np.array(bin_acc)[valid] - np.array(bin_conf)[valid])
    )

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(np.array(bin_acc)[valid] - np.array(bin_conf)[valid]))

    # Negative Log Likelihood (NLL) for binary case
    nll = -np.mean(
        true_labels * np.log(predictions) +
        (1 - true_labels) * np.log(1 - predictions)
    )

    # Brier Score for binary case
    brier = np.mean((predictions - true_labels) ** 2)

    return bin_acc, bin_centers, ece, mce, nll, brier


if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    configs = {
        "params": {
            # "epochs": 1000,
            "epochs": 100,
            "batch_size": 100,
            "init_lr": 7e-4,
            "lr_reduce_factor": 0.5,
            "lr_schedule_patience": 25,
            "min_lr": 1e-4,
            "weight_decay": 0.0,
            "print_epoch_interval": 5,
        }
    }

    params = configs['params']
    num_layers = 3

    dataset = MIMICDataset(root=pjoin(cwd, 'data', 'MIMICDataset'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # Count classes
    labels = dataset.data.y.numpy()  # Convert to numpy array
    class_counts = Counter(labels)

    # Print counts
    label_title = {0: 'No Admission', 1: 'Admitted', 2: 'ICU Admission'}

    print(f'Device: {device}')
    print()

    min_val = np.inf
    majority_class = 0
    minority_class = 1
    for label, count in class_counts.items():
        if label in [0, 1]:  # Not considering the OOD data
            if min_val > count:
                min_val = count
                majority_class = minority_class
                minority_class = label

    ood_target = 2
    num_classes = dataset.num_classes - 1

    iid_data_idx = []
    iid_data_y = []
    ood_data_idx = []
    for idx, data in enumerate(dataset):
        if data.y.item() != ood_target:
            iid_data_idx.append(idx)
            iid_data_y.append(data.y.item())
        else:
            ood_data_idx.append(idx)

    all_iid_data = np.array(iid_data_idx)
    all_labels = np.array(iid_data_y)

    iid_df = pd.DataFrame({'idx': all_iid_data, 'label': all_labels})

    # Split into majority and minority classes
    majority = iid_df[iid_df['label'] == majority_class]
    minority = iid_df[iid_df['label'] == minority_class]

    print(f'Down Sampling...')
    print()
    # Downsample majority class to the size of the minority class
    majority_downsampled = resample(
        majority,
        replace=False,  # No replacement
        n_samples=len(minority),  # Match minority class size
        random_state=42  # For reproducibility
    )

    # Combine the undersampled majority class with the minority class
    undersampled_data = pd.concat([majority_downsampled, minority])

    # Shuffle the result
    undersampled_data = undersampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    iid_data = undersampled_data['idx'].values
    labels = undersampled_data['label'].values

    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    split_number = 0

    ensemble_report = {'Approach': 'Ensemble'}
    self_dist_report = {'Approach': 'Self-Distillation'}
    single_model_report = {'Approach': 'Single Model'}
    mc_dropout_report = {'Approach': 'MCDropout'}

    model_name = 'graph_conv'

    print('Computing Metrics')
    print()
    ##########################################################################################
    # Ensemble: Time to Train
    # Open and read the time file for ensemble
    times = []
    for split_number in range(1, 6):
        times.append(0)
        for ensemble_num in range(1, 4):
            time_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                              f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_time_to_train.txt')
            with open(time_file, "r") as file:
                # Reading from a file
                times[split_number - 1] += float(file.read().split(" ")[-1])
    times = np.array(times)
    ensemble_report['Time to Train'] = "{:.2f} \u00B1 {:.2f}".format(times.mean(), times.std())
    # -------------------------------------------------------------------------------------------------
    # Single Model: Time to Train
    # Open and read the time file for ensemble
    times = []
    for split_number in range(1, 6):
        ensemble_num = 1
        time_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                          f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_time_to_train.txt')
        with open(time_file, "r") as file:
            # Reading from a file
            times.append(float(file.read().split(" ")[-1]))
    times = np.array(times)
    single_model_report['Time to Train'] = "{:.2f} \u00B1 {:.2f}".format(times.mean(), times.std())
    # -------------------------------------------------------------------------------------------------
    # MC Dropout: Time to Train
    # Open and read the time file for ensemble
    times = []
    for split_number in range(1, 6):
        time_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                          f'{model_name}_mc_dropout_split_{split_number}_time_to_train.txt')
        with open(time_file, "r") as file:
            # Reading from a file
            times.append(float(file.read().split(" ")[-1]))
    times = np.array(times)
    mc_dropout_report['Time to Train'] = "{:.2f} \u00B1 {:.2f}".format(times.mean(), times.std())
    # -------------------------------------------------------------------------------------------------
    # Self-Distillation: Time to Train
    # Open and read the time file for ensemble
    times = []
    for split_number in range(1, 6):
        time_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                          f'{model_name}_split_{split_number}_time_to_train.txt')
        with open(time_file, "r") as file:
            # Reading from a file
            times.append(float(file.read().split(" ")[-1]))
    times = np.array(times)
    self_dist_report['Time to Train'] = "{:.2f} \u00B1 {:.2f}".format(times.mean(), times.std())
    ##########################################################################################
    # Ensemble: Performance
    split_values = {
        # 'probabilities': [],
        # 'mean_probabilities': [],
        'roc_auc': [],
        'accuracies': [],
        'f1_scores': [],
    }
    for split_number in range(1, 6):
        all_predictions = []
        all_probabilities = []
        for ensemble_num in range(1, 4):
            scores_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                                f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_test_logit_vals.txt')
            scores = np.loadtxt(scores_file, dtype="double")
            all_probabilities.append(F.softmax(torch.tensor(scores).double(), dim=-1).numpy())
            all_predictions.append(torch.tensor(scores).double().argmax(dim=1).numpy().tolist())

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        # majority_votes = np.apply_along_axis(majority_vote, axis=1, arr=np.array(all_predictions).T)
        soft_votes = np.argmax(np.mean(all_probabilities, axis=0), axis=1)
        true_labels_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                                 f'{model_name}_ensemble_1_split_{split_number}_test_true_classes.txt')
        true_labels = np.loadtxt(true_labels_file, dtype="double")
        # split_values['probabilities'].append(all_probabilities)
        # split_values['mean_probabilities'].append(np.mean(all_probabilities, axis=0))
        split_values['roc_auc'].append(
            roc_auc_score(true_labels, np.mean(all_probabilities, axis=0)[:, 1]))
        split_values['accuracies'].append(accuracy_score(true_labels, soft_votes))
        split_values['f1_scores'].append(f1_score(true_labels, soft_votes))

    split_values['roc_auc'] = np.array(split_values['roc_auc'])
    split_values['accuracies'] = np.array(split_values['accuracies'])
    split_values['f1_scores'] = np.array(split_values['f1_scores'])
    ensemble_report['Accuracy'] = "{:.2f} \u00B1 {:.2f}".format(split_values['accuracies'].mean(),
                                                                split_values['accuracies'].std())
    ensemble_report['F1 Score'] = "{:.2f} \u00B1 {:.2f}".format(split_values['f1_scores'].mean(),
                                                                split_values['f1_scores'].std())
    ensemble_report['ROC AUC'] = "{:.2f} \u00B1 {:.2f}".format(split_values['roc_auc'].mean(),
                                                               split_values['roc_auc'].std())
    # -------------------------------------------------------------------------------------------------
    # Single Model: Performance
    split_values = {
        # 'probabilities': [],
        # 'mean_probabilities': [],
        'roc_auc': [],
        'accuracies': [],
        'f1_scores': [],
    }
    for split_number in range(1, 6):
        all_predictions = []
        all_probabilities = []
        ensemble_num = 1
        scores_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                            f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_test_logit_vals.txt')
        scores = np.loadtxt(scores_file, dtype="double")
        all_probabilities = F.softmax(torch.tensor(scores).double(), dim=-1).numpy()
        all_predictions = torch.tensor(scores).double().argmax(dim=1).numpy().tolist()

        true_labels_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                                 f'{model_name}_ensemble_1_split_{split_number}_test_true_classes.txt')
        true_labels = np.loadtxt(true_labels_file, dtype="double")
        # split_values['probabilities'].append(all_probabilities)
        # split_values['mean_probabilities'].append(np.mean(all_probabilities, axis=0))
        split_values['roc_auc'].append(
            roc_auc_score(true_labels, all_probabilities[:, 1]))
        split_values['accuracies'].append(accuracy_score(true_labels, all_predictions))
        split_values['f1_scores'].append(f1_score(true_labels, all_predictions))

    split_values['roc_auc'] = np.array(split_values['roc_auc'])
    split_values['accuracies'] = np.array(split_values['accuracies'])
    split_values['f1_scores'] = np.array(split_values['f1_scores'])
    single_model_report['Accuracy'] = "{:.2f} \u00B1 {:.2f}".format(split_values['accuracies'].mean(),
                                                                    split_values['accuracies'].std())
    single_model_report['F1 Score'] = "{:.2f} \u00B1 {:.2f}".format(split_values['f1_scores'].mean(),
                                                                    split_values['f1_scores'].std())
    single_model_report['ROC AUC'] = "{:.2f} \u00B1 {:.2f}".format(split_values['roc_auc'].mean(),
                                                                   split_values['roc_auc'].std())
    # -------------------------------------------------------------------------------------------------
    # MCDropout: Performance
    split_values = {
        # 'probabilities': [],
        # 'mean_probabilities': [],
        'roc_auc': [],
        'accuracies': [],
        'f1_scores': [],
    }
    for split_number in range(1, 6):
        all_predictions = []
        all_probabilities = []
        all_probabilities_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                                       f'{model_name}_mc_dropout_split_{split_number}_test_probabilities_vals.txt')
        all_probabilities = np.loadtxt(all_probabilities_file, dtype="double")
        all_predictions_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                                     f'{model_name}_mc_dropout_split_{split_number}_test_predicted_classes.txt')
        all_predictions = np.loadtxt(all_predictions_file, dtype="double").tolist()

        true_labels_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                                 f'{model_name}_mc_dropout_split_{split_number}_test_true_classes.txt')
        true_labels = np.loadtxt(true_labels_file, dtype="double")
        # split_values['probabilities'].append(all_probabilities)
        # split_values['mean_probabilities'].append(np.mean(all_probabilities, axis=0))
        split_values['roc_auc'].append(
            roc_auc_score(true_labels, all_probabilities[:, 1]))
        split_values['accuracies'].append(accuracy_score(true_labels, all_predictions))
        split_values['f1_scores'].append(f1_score(true_labels, all_predictions))

    split_values['roc_auc'] = np.array(split_values['roc_auc'])
    split_values['accuracies'] = np.array(split_values['accuracies'])
    split_values['f1_scores'] = np.array(split_values['f1_scores'])
    mc_dropout_report['Accuracy'] = "{:.2f} \u00B1 {:.2f}".format(split_values['accuracies'].mean(),
                                                                  split_values['accuracies'].std())
    mc_dropout_report['F1 Score'] = "{:.2f} \u00B1 {:.2f}".format(split_values['f1_scores'].mean(),
                                                                  split_values['f1_scores'].std())
    mc_dropout_report['ROC AUC'] = "{:.2f} \u00B1 {:.2f}".format(split_values['roc_auc'].mean(),
                                                                 split_values['roc_auc'].std())
    # -------------------------------------------------------------------------------------------------
    # Self-Distillation: Performance
    split_values = {
        # 'probabilities': [],
        'roc_auc': [],
        'accuracies': [],
        'f1_scores': [],
    }
    for split_number in range(1, 6):
        all_predictions = []
        all_probabilities = []
        for layer in range(1, 4):
            scores_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                f'{model_name}_layer_{layer}_split_{split_number}_test_logit_vals.txt')
            scores = np.loadtxt(scores_file, dtype="double")
            all_probabilities.append(F.softmax(torch.tensor(scores).double(), dim=-1).numpy())
            all_predictions.append(torch.tensor(scores).double().argmax(dim=1).numpy().tolist())

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        true_labels_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                 f'{model_name}_layer_1_split_{split_number}_test_true_classes.txt')
        true_labels = np.loadtxt(true_labels_file, dtype="double")
        # split_values['probabilities'].append(all_probabilities[-1])
        split_values['roc_auc'].append(
            roc_auc_score(true_labels, all_probabilities[-1][:, 1]))
        split_values['accuracies'].append(accuracy_score(true_labels, all_predictions[-1]))
        split_values['f1_scores'].append(f1_score(true_labels, all_predictions[-1]))

    split_values['roc_auc'] = np.array(split_values['roc_auc'])
    split_values['accuracies'] = np.array(split_values['accuracies'])
    split_values['f1_scores'] = np.array(split_values['f1_scores'])
    self_dist_report['Accuracy'] = "{:.2f} \u00B1 {:.2f}".format(split_values['accuracies'].mean(),
                                                                 split_values['accuracies'].std())
    self_dist_report['F1 Score'] = "{:.2f} \u00B1 {:.2f}".format(split_values['f1_scores'].mean(),
                                                                 split_values['f1_scores'].std())
    self_dist_report['ROC AUC'] = "{:.2f} \u00B1 {:.2f}".format(split_values['roc_auc'].mean(),
                                                                split_values['roc_auc'].std())
    ##########################################################################################
    # Ensemble & Single Model: Number of Parameters and Test Time
    total_params = 0
    single_model_params = 0
    inference_times = []
    single_model_times = []
    split_number = 0
    for _, test_index in skf.split(iid_data, labels):
        split_number += 1
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)
        ensemble_params = []
        ensemble_times = []
        for ensemble_num in range(1, 4):
            test_dataset_idx = iid_data[test_index]
            test_y = labels[test_index]

            test_dataset = dataset.index_select(test_dataset_idx)

            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                     collate_fn=test_dataset.collate)

            _, edge_types = dataset[0].metadata()
            metadata = {
                'edges': edge_types,
                'num_features': dataset.num_features,
            }
            model = GraphConvGNN(metadata, num_classes, device, 128, num_layers=num_layers, dropout=0.5)
            model = model.to(device)

            root_ckpt_dir = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic')
            ckpt_dir = os.path.join(root_ckpt_dir, f"Ensemble_{ensemble_num}_RUN_" + str(split_number))
            model.load_state_dict(
                torch.load('{}.pkl'.format(pjoin(ckpt_dir, f"epoch_{str(params['epochs'] - 1)}")), weights_only=True))
            model.eval()

            single_model_params = count_parameters(model)
            ensemble_params.append(single_model_params)

            # To make sure lazy initialization is not affecting
            for data in test_loader:
                data = data.to(device)
                batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
                output = model(data.x_dict, data.edge_index_dict, batch_dict)
                break
            if torch.cuda.is_available():
                # Create CUDA events
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                # Record the start event
                start.record()
                for data in test_loader:
                    data = data.to(device)
                    batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
                    output = model(data.x_dict, data.edge_index_dict, batch_dict)
                # Record the end event
                end.record()

                # Synchronize the GPU
                torch.cuda.synchronize()

                # Calculate the elapsed time
                elapsed_time_ms = start.elapsed_time(end)
                elapsed_time_s = elapsed_time_ms / 1000.0
                ensemble_times.append(elapsed_time_s)
                if ensemble_num == 1:
                    single_model_times.append(elapsed_time_s)
            else:
                start_time = time.time()
                for data in test_loader:
                    data = data.to(device)
                    batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
                    output = model(data.x_dict, data.edge_index_dict, batch_dict)
                end_time = time.time()
                elapsed_time_s = end_time - start_time
                ensemble_times.append(elapsed_time_s)
                if ensemble_num == 1:
                    single_model_times.append(elapsed_time_s)

        total_params = np.array(ensemble_params).sum()
        inference_times.append(np.array(ensemble_times).sum())
    ensemble_report['Inference Time'] = "{:.2f} \u00B1 {:.2f}".format(np.array(inference_times).mean(),
                                                                      np.array(inference_times).std())
    ensemble_report['# Params'] = total_params
    single_model_report['Inference Time'] = "{:.2f} \u00B1 {:.2f}".format(np.array(single_model_times).mean(),
                                                                          np.array(single_model_times).std())
    single_model_report['# Params'] = single_model_params
    # -------------------------------------------------------------------------------------------------
    # MCDropout: Number of Parameters and Test Time
    total_params = 0
    inference_times = []
    split_number = 0
    for _, test_index in skf.split(iid_data, labels):
        split_number += 1
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)

        test_dataset_idx = iid_data[test_index]
        test_y = labels[test_index]

        test_dataset = dataset.index_select(test_dataset_idx)

        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                 collate_fn=test_dataset.collate)

        _, edge_types = dataset[0].metadata()
        metadata = {
            'edges': edge_types,
            'num_features': dataset.num_features,
        }
        model = MCDropoutGraphConvGNN(metadata, num_classes, device, 128, num_layers=num_layers, dropout=0.5)
        model = model.to(device)

        root_ckpt_dir = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic')
        ckpt_dir = os.path.join(root_ckpt_dir, f"mc_dropout_RUN_" + str(split_number))
        model.load_state_dict(
            torch.load('{}.pkl'.format(pjoin(ckpt_dir, f"epoch_{str(params['epochs'] - 1)}")), weights_only=True))
        model.eval()

        total_params = count_parameters(model)

        # To make sure lazy initialization is not affecting
        for data in test_loader:
            data = data.to(device)
            n_samples = 100
            outs = mc_inference(model, data, n_samples=n_samples)
            break
        if torch.cuda.is_available():
            # Create CUDA events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # Record the start event
            start.record()
            for data in test_loader:
                data = data.to(device)
                n_samples = 100
                outs = mc_inference(model, data, n_samples=n_samples)
            # Record the end event
            end.record()

            # Synchronize the GPU
            torch.cuda.synchronize()

            # Calculate the elapsed time
            elapsed_time_ms = start.elapsed_time(end)
            elapsed_time_s = elapsed_time_ms / 1000.0
            inference_times.append(elapsed_time_s)
        else:
            start_time = time.time()
            for data in test_loader:
                data = data.to(device)
                n_samples = 100
                outs = mc_inference(model, data, n_samples=n_samples)
            end_time = time.time()
            elapsed_time_s = end_time - start_time
            inference_times.append(elapsed_time_s)

    mc_dropout_report['Inference Time'] = "{:.2f} \u00B1 {:.2f}".format(np.array(inference_times).mean(),
                                                                        np.array(inference_times).std())
    mc_dropout_report['# Params'] = total_params
    # -------------------------------------------------------------------------------------------------
    # Self-Distillation: Number of Parameters and Test Time
    total_params = 0
    inference_times = []
    split_number = 0
    for _, test_index in skf.split(iid_data, labels):
        split_number += 1
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)
        test_dataset_idx = iid_data[test_index]
        test_y = labels[test_index]

        test_dataset = dataset.index_select(test_dataset_idx)

        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                 collate_fn=test_dataset.collate)

        _, edge_types = dataset[0].metadata()
        metadata = {
            'edges': edge_types,
            'num_features': dataset.num_features,
        }
        model = SelfDistillationGraphConvGNN(metadata, num_classes, device, 128, num_layers=num_layers, dropout=0.5)
        model = model.to(device)

        root_ckpt_dir = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic')
        ckpt_dir = os.path.join(root_ckpt_dir, f"Self_Distillation_RUN_" + str(split_number))
        model.load_state_dict(
            torch.load('{}.pkl'.format(pjoin(ckpt_dir, f"epoch_{str(params['epochs'] - 1)}")), weights_only=True))
        model.eval()

        # Calculate and print the parameter count
        total_params = count_parameters(model)

        # To make sure lazy initialization is not affecting
        for data in test_loader:
            data = data.to(device)
            batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
            output_list, _ = model(data.x_dict, data.edge_index_dict, batch_dict)
            break
        if torch.cuda.is_available():
            # Create CUDA events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # Record the start event
            start.record()
            for data in test_loader:
                data = data.to(device)
                batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
                output_list, _ = model(data.x_dict, data.edge_index_dict, batch_dict)
            # Record the end event
            end.record()

            # Synchronize the GPU
            torch.cuda.synchronize()

            # Calculate the elapsed time
            elapsed_time_ms = start.elapsed_time(end)
            elapsed_time_s = elapsed_time_ms / 1000.0
            inference_times.append(elapsed_time_s)
        else:
            start_time = time.time()
            for data in test_loader:
                data = data.to(device)
                batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
                output_list, _ = model(data.x_dict, data.edge_index_dict, batch_dict)
            end_time = time.time()
            elapsed_time_s = end_time - start_time
            inference_times.append(elapsed_time_s)

    self_dist_report['Inference Time'] = "{:.2f} \u00B1 {:.2f}".format(np.array(inference_times).mean(),
                                                                       np.array(inference_times).std())
    self_dist_report['# Params'] = total_params
    ##########################################################################################
    # Ensemble: ECE, MCE, BrierScore, NLL
    ens_metrics = {
        'ece': [],
        'mce': [],
        'brier': [],
        'nll': [],
    }
    for split_number in range(1, 6):
        all_probs = [[] for i in range(0, 3)]
        for ensemble_num in range(1, 4):
            scores_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                                f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_test_logit_vals.txt')
            scores = np.loadtxt(scores_file, dtype="double")
            calc_probs = F.softmax(torch.tensor(scores).double(), dim=-1)
            calc_probs = calc_probs / calc_probs.sum(dim=-1, keepdim=True)
            calc_smoothed_probs = smooth_probabilities(calc_probs)
            all_probs[ensemble_num - 1] = calc_smoothed_probs.numpy().tolist()
            # all_probs[ensemble_num - 1] = F.softmax(torch.tensor(scores).double(), dim=-1).numpy().tolist()

        ref_probs = np.mean(np.array(all_probs), axis=0)

        true_classes_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                                  f'{model_name}_ensemble_1_split_{split_number}_test_true_classes.txt')
        true_classes = np.loadtxt(true_classes_file, dtype="double").astype(int)

        _, _, ece, mce, nll, brier = compute_calibration_metrics(
            ref_probs, true_classes.tolist()
        )
        ens_metrics['ece'].append(ece)
        ens_metrics['mce'].append(mce)
        ens_metrics['nll'].append(nll)
        ens_metrics['brier'].append(brier)
    for metric in ens_metrics:
        vals = np.array(ens_metrics[metric])
        ensemble_report[metric.upper()] = "{:.2f} \u00B1 {:.2f}".format(vals.mean(), vals.std())
    # -------------------------------------------------------------------------------------------------
    # Single Model: ECE, MCE, BrierScore, NLL
    # Nothing to compute
    single_model_report['ECE'] = '-'
    single_model_report['MCE'] = '-'
    single_model_report['BRIER'] = '-'
    single_model_report['NLL'] = '-'
    # -------------------------------------------------------------------------------------------------
    # MCDropout: ECE, MCE, BrierScore, NLL
    mcd_metrics = {
        'ece': [],
        'mce': [],
        'brier': [],
        'nll': [],
    }
    for split_number in range(1, 6):
        ref_probs_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                               f'{model_name}_mc_dropout_split_{split_number}_test_probabilities_vals.txt')
        ref_probs = np.loadtxt(ref_probs_file, dtype="double")
        smoothed_ref_probs = smooth_probabilities_np(ref_probs)
        # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()

        true_classes_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                                  f'{model_name}_mc_dropout_split_{split_number}_test_true_classes.txt')
        true_classes = np.loadtxt(true_classes_file, dtype="double").astype(int)

        _, _, ece, mce, nll, brier = compute_calibration_metrics(
            smoothed_ref_probs, true_classes.tolist()
        )
        mcd_metrics['ece'].append(ece)
        mcd_metrics['mce'].append(mce)
        mcd_metrics['nll'].append(nll)
        mcd_metrics['brier'].append(brier)

    for metric in mcd_metrics:
        vals = np.array(mcd_metrics[metric])
        mc_dropout_report[metric.upper()] = "{:.2f} \u00B1 {:.2f}".format(vals.mean(), vals.std())
    # -------------------------------------------------------------------------------------------------
    # Self-Distillation: ECE, MCE, BrierScore, NLL
    self_distillation_metrics = {
        'ece': [],
        'mce': [],
        'brier': [],
        'nll': [],
    }
    for split_number in range(1, 6):
        ref_scores_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                f'{model_name}_layer_{num_layers}_split_{split_number}_test_logit_vals.txt')
        ref_scores = np.loadtxt(ref_scores_file, dtype="double")
        ref_probs = F.softmax(torch.tensor(ref_scores).double(), dim=-1)
        ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
        smoothed_ref_probs = smooth_probabilities(ref_probs)
        # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()

        true_classes_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                  f'{model_name}_layer_{num_layers}_split_{split_number}_test_true_classes.txt')
        true_classes = np.loadtxt(true_classes_file, dtype="double").astype(int)

        _, _, ece, mce, nll, brier = compute_calibration_metrics(
            smoothed_ref_probs.numpy(), true_classes.tolist()
        )
        self_distillation_metrics['ece'].append(ece)
        self_distillation_metrics['mce'].append(mce)
        self_distillation_metrics['nll'].append(nll)
        self_distillation_metrics['brier'].append(brier)

    for metric in self_distillation_metrics:
        vals = np.array(self_distillation_metrics[metric])
        self_dist_report[metric.upper()] = "{:.2f} \u00B1 {:.2f}".format(vals.mean(), vals.std())
    ##########################################################################################
    print(tabulate(pd.DataFrame([single_model_report, mc_dropout_report, ensemble_report, self_dist_report]).iloc[:,
                   [0, 2, 3, 4, 6, 1, 5, 7, 8, 9, 10]], headers='keys',
                   tablefmt='psql', showindex=False))
