import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import torch.nn.functional as F
from utils.metrics import jensenshannon_metric, weighted_agreement, linear_weight, nonlinear_weight


def smooth_probabilities(predictions, epsilon=1e-8):
    # Add epsilon to all probabilities
    smoothed = predictions + epsilon
    # Renormalize along the class dimension (last dimension)
    smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True)
    return smoothed


if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    ################### Self-Distillation Model ######################
    agreement_vals = []
    linear_agreement_vals = []
    nonlinear_agreement_vals = []
    for split_number in range(1, 6):
        model_name = 'graph_conv'
        ref_scores_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                f'{model_name}_layer_3_split_{split_number}_test_logit_vals.txt')
        ref_scores = np.loadtxt(ref_scores_file, dtype="double")
        ref_probs = F.softmax(torch.tensor(ref_scores).double(), dim=-1)
        ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
        smoothed_ref_probs = smooth_probabilities(ref_probs)
        # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()
        # ref_predictions = torch.tensor(ref_scores).argmax(dim=1).tolist()
        ref_predictions = torch.argmax(smoothed_ref_probs, dim=-1).tolist()
        layers_jsd = [[] for _ in range(1, 3)]
        predictions = [[] for _ in range(1, 3)]
        for layer in range(1, 3):
            scores_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                f'{model_name}_layer_{layer}_split_{split_number}_test_logit_vals.txt')
            scores = np.loadtxt(scores_file, dtype="double")
            probs = F.softmax(torch.tensor(scores).double(), dim=-1)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            smoothed_probs = smooth_probabilities(probs)
            # probs = F.softmax(torch.tensor(scores).double(), dim=-1).numpy().tolist()
            # predictions[layer - 1] = torch.tensor(scores).double().argmax(dim=1).tolist()
            predictions[layer - 1] = torch.argmax(smoothed_probs, dim=-1).tolist()
            # for idx in range(len(ref_scores)):
            for idx in range(len(smoothed_probs)):
                jsd_value = jensenshannon_metric(smoothed_probs[idx], smoothed_ref_probs[idx])
                # jsd_value = jensenshannon_metric(ref_probs[idx], probs[idx])
                layers_jsd[layer - 1].append(jsd_value)

        layers_weight = [[] for _ in range(1, 3)]
        for layer in range(1, 3):
            layers_weight[layer - 1] = linear_weight(3, 3 - layer, predictions[layer - 1], ref_predictions)
        test_agreement_values = weighted_agreement(layers_weight, layers_jsd, normalize=False,
                                                   num_layers=3, weight_func_type='linear')
        linear_agreement_vals += test_agreement_values

        layers_weight = [[] for _ in range(1, 3)]
        for layer in range(1, 3):
            layers_weight[layer - 1] = nonlinear_weight(3, 3 - layer, predictions[layer - 1], ref_predictions)
        test_agreement_values = weighted_agreement(layers_weight, layers_jsd, normalize=False,
                                                   num_layers=3, weight_func_type='nonlinear')
        nonlinear_agreement_vals += test_agreement_values

    linear_agreement_vals = np.array(linear_agreement_vals)
    nonlinear_agreement_vals = np.array(nonlinear_agreement_vals)

    # Plot the KDE in Color
    plt.figure(figsize=(10, 6))
    sns.kdeplot(linear_agreement_vals, label='UC Metric (Linear Weight)', shade=True)
    sns.kdeplot(nonlinear_agreement_vals, label='UC Metric (Nonlinear Weight)', shade=True)

    # Add labels and legend
    plt.xlabel('Metric Value')
    plt.ylabel('Density')
    plt.title('KDE Plot of the UC Metrics (Not Normalized)')
    plt.legend()

    try:
        os.makedirs(pjoin(cwd, 'data', 'figs'))
    except FileExistsError:
        # directory already exists
        pass

    plt.savefig(pjoin(cwd, 'data', 'figs', f'mimic_density_self_jsd_metrics.pdf'))
    plt.close()

    # Plot the KDE in Gray-Scale
    plt.figure(figsize=(10, 6))
    sns.kdeplot(linear_agreement_vals, label='UC Metric (Linear Weight)', shade=True, color='gray')
    sns.kdeplot(nonlinear_agreement_vals, label='UC Metric (Nonlinear Weight)', shade=True, color='black')

    # Add labels and legend
    plt.xlabel('Metric Value')
    plt.ylabel('Density')
    plt.title('KDE Plot of the UC Metrics (Not Normalized)')
    plt.legend()

    try:
        os.makedirs(pjoin(cwd, 'data', 'figs'))
    except FileExistsError:
        # directory already exists
        pass

    plt.savefig(pjoin(cwd, 'data', 'figs', f'mimic_density_self_jsd_metrics-gray.pdf'))
