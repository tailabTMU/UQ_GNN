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
    value_idx = []
    lin_values = []
    nonlin_values = []
    for alpha_val in [0.1, 0.4, 0.6, 0.8]:
        agreement_vals = []
        linear_agreement_vals = []
        nonlinear_agreement_vals = []
        for split_number in range(1, 6):
            model_name = 'sage'
            ref_scores_file = pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes',
                                    f'{model_name}_layer_4_split_{split_number}_alpha_{alpha_val}test_logit_vals.txt')
            ref_scores = np.loadtxt(ref_scores_file, dtype="double")
            ref_probs = F.softmax(torch.tensor(ref_scores).double(), dim=-1)
            ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
            smoothed_ref_probs = smooth_probabilities(ref_probs)
            # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()
            # ref_predictions = torch.tensor(ref_scores).argmax(dim=1).tolist()
            ref_predictions = torch.argmax(smoothed_ref_probs, dim=-1).tolist()
            layers_jsd = [[] for _ in range(1, 4)]
            predictions = [[] for _ in range(1, 4)]
            for layer in range(1, 4):
                scores_file = pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes',
                                    f'{model_name}_layer_{layer}_split_{split_number}_alpha_{alpha_val}test_logit_vals.txt')
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

            layers_weight = [[] for _ in range(1, 4)]
            for layer in range(1, 4):
                layers_weight[layer - 1] = linear_weight(4, 4 - layer, predictions[layer - 1], ref_predictions)
            test_agreement_values = weighted_agreement(layers_weight, layers_jsd, normalize=False,
                                                       num_layers=4, weight_func_type='linear')
            linear_agreement_vals += test_agreement_values

            layers_weight = [[] for _ in range(1, 4)]
            for layer in range(1, 4):
                layers_weight[layer - 1] = nonlinear_weight(4, 4 - layer, predictions[layer - 1], ref_predictions)
            test_agreement_values = weighted_agreement(layers_weight, layers_jsd, normalize=False,
                                                       num_layers=4, weight_func_type='nonlinear')
            nonlinear_agreement_vals += test_agreement_values

        value_idx.append(~np.isnan(np.array(linear_agreement_vals)))
        lin_values.append(np.array(linear_agreement_vals))
        nonlin_values.append(np.array(nonlinear_agreement_vals))

    result = np.logical_and.reduce(value_idx)
    for idx, alpha_val in enumerate([0.1, 0.4, 0.6, 0.8]):
        linear_agreement_vals = lin_values[idx]
        nonlinear_agreement_vals = nonlin_values[idx]
        linear_agreement_vals = linear_agreement_vals[result]
        nonlinear_agreement_vals = nonlinear_agreement_vals[result]
        print(f'Alpha {alpha_val}')
        print('Linear Agreement: {:.2f}'.format(linear_agreement_vals.mean()))
        print('Nonlinear Agreement: {:.2f}'.format(nonlinear_agreement_vals.mean()))
        print('==================')
