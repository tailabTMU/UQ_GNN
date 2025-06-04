import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import torch.nn.functional as F


def entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities), axis=1)


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


if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    self_dist_ood_prediction_values = []
    self_dist_test_prediction_values = []
    mc_dropout_ood_prediction_values = []
    mc_dropout_test_prediction_values = []
    ens_ood_prediction_values = []
    ens_test_prediction_values = []
    for split_number in range(1, 6):
        ################### Self-Distillation Model ######################
        model_name = 'graph_conv'
        ref_scores_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                f'{model_name}_layer_3_split_{split_number}_test_logit_vals.txt')
        ref_scores = np.loadtxt(ref_scores_file, dtype="double")
        ref_probs = F.softmax(torch.tensor(ref_scores).double(), dim=-1)
        ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
        smoothed_ref_probs = smooth_probabilities(ref_probs)
        # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()

        self_dist_test_prediction_values += smoothed_ref_probs.numpy().tolist()

        ref_scores_file = pjoin(cwd, 'saved_info_graph_classification', 'self_distillation_mimic',
                                f'{model_name}_layer_3_split_{split_number}_ood_logit_vals.txt')
        ref_scores = np.loadtxt(ref_scores_file, dtype="double")
        ref_probs = F.softmax(torch.tensor(ref_scores).double(), dim=-1)
        ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
        smoothed_ref_probs = smooth_probabilities(ref_probs)
        # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()
        self_dist_ood_prediction_values.append(smoothed_ref_probs.numpy().tolist())

        ################### MCDropout Model ######################
        model_name = 'graph_conv'
        ref_probs_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                                f'{model_name}_mc_dropout_split_{split_number}_test_probabilities_vals.txt')
        ref_probs = np.loadtxt(ref_probs_file, dtype="double")
        ref_probs = ref_probs / ref_probs.sum(axis=-1, keepdims=True)
        smoothed_ref_probs = smooth_probabilities_np(ref_probs)
        # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()

        mc_dropout_test_prediction_values += smoothed_ref_probs.tolist()

        ref_probs_file = pjoin(cwd, 'saved_info_graph_classification', 'mc_dropout_mimic',
                                f'{model_name}_mc_dropout_split_{split_number}_ood_probabilities_vals.txt')
        ref_probs = np.loadtxt(ref_probs_file, dtype="double")
        ref_probs = ref_probs / ref_probs.sum(axis=-1, keepdims=True)
        smoothed_ref_probs = smooth_probabilities_np(ref_probs)
        # ref_probs = F.softmax(torch.tensor(ref_scores), dim=-1).numpy().tolist()
        mc_dropout_ood_prediction_values.append(smoothed_ref_probs.tolist())

        ################### Ensemble Model ######################

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
        ens_test_prediction_values += ref_probs.tolist()

        all_probs = [[] for i in range(0, 3)]
        for ensemble_num in range(1, 4):
            scores_file = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_mimic',
                                f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_ood_logit_vals.txt')
            scores = np.loadtxt(scores_file, dtype="double")
            calc_probs = F.softmax(torch.tensor(scores).double(), dim=-1)
            calc_probs = calc_probs / calc_probs.sum(dim=-1, keepdim=True)
            calc_smoothed_probs = smooth_probabilities(calc_probs)
            all_probs[ensemble_num - 1] = calc_smoothed_probs.numpy().tolist()
            # all_probs[ensemble_num - 1] = F.softmax(torch.tensor(scores).double(), dim=-1).numpy().tolist()

        ref_probs = np.mean(np.array(all_probs), axis=0)
        ens_ood_prediction_values.append(ref_probs.tolist())

    self_dist_ood_mean_prediction_probabilities = np.array(self_dist_ood_prediction_values).mean(axis=0).tolist()
    mc_dropout_ood_mean_prediction_probabilities = np.array(mc_dropout_ood_prediction_values).mean(axis=0).tolist()
    ens_ood_mean_prediction_probabilities = np.array(ens_ood_prediction_values).mean(axis=0).tolist()
    self_dist_entropy_ood = entropy(self_dist_ood_mean_prediction_probabilities)
    mc_dropout_entropy_ood = entropy(mc_dropout_ood_mean_prediction_probabilities)
    ens_entropy_ood = entropy(ens_ood_mean_prediction_probabilities)
    self_dist_entropy_test = entropy(self_dist_test_prediction_values)
    mc_dropout_entropy_test = entropy(mc_dropout_test_prediction_values)
    ens_entropy_test = entropy(ens_test_prediction_values)

    # Create the plot in Color
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Row 1, Column 1 - Self-Distillation (Blue)
    sns.kdeplot(self_dist_entropy_test, color='blue', ax=axes[0, 0])
    axes[0, 0].set_title("ID Data", fontsize=21)
    axes[0, 0].set_xlabel("Entropy Values", fontsize=21)
    axes[0, 0].set_ylabel("Density", fontsize=21)

    # Row 1, Column 1 - MCDropout (Blue)
    sns.kdeplot(mc_dropout_entropy_test, color='blue', ax=axes[0, 2])
    axes[0, 2].set_title("ID Data", fontsize=21)
    axes[0, 2].set_xlabel("Entropy Values", fontsize=21)
    axes[0, 2].set_ylabel("Density", fontsize=21)

    # Row 1, Column 2 - Ensemble (Blue)
    sns.kdeplot(ens_entropy_test, color='blue', ax=axes[0, 1])
    axes[0, 1].set_title("ID Data", fontsize=21)
    axes[0, 1].set_xlabel("Entropy Values", fontsize=21)
    axes[0, 1].set_ylabel("Density", fontsize=21)

    # Row 2, Column 1 - Self-Distillation (Red)
    sns.kdeplot(self_dist_entropy_ood, color='red', ax=axes[1, 0])
    axes[1, 0].set_title("OOD Data", fontsize=21)
    axes[1, 0].set_xlabel("Entropy Values", fontsize=21)
    axes[1, 0].set_ylabel("Density", fontsize=21)

    # Row 2, Column 1 - MCDRopout (Red)
    sns.kdeplot(mc_dropout_entropy_ood, color='red', ax=axes[1, 2])
    axes[1, 2].set_title("OOD Data", fontsize=21)
    axes[1, 2].set_xlabel("Entropy Values", fontsize=21)
    axes[1, 2].set_ylabel("Density", fontsize=21)

    # Row 2, Column 2 - Ensemble (Red)
    sns.kdeplot(ens_entropy_ood, color='red', ax=axes[1, 1])
    axes[1, 1].set_title("OOD Data", fontsize=21)
    axes[1, 1].set_xlabel("Entropy Values", fontsize=21)
    axes[1, 1].set_ylabel("Density", fontsize=21)

    # Set main titles for columns
    axes[0, 0].set_ylabel("Density", fontsize=21)
    axes[1, 0].set_ylabel("Density", fontsize=21)

    for ax in axes.flatten():
        # Dynamically set 6 ticks using np.linspace()
        _, x_max = ax.get_xlim()  # Get X-axis range
        _, y_max = ax.get_ylim()  # Get Y-axis range

        ax.set_xticks(np.round(np.linspace(0, x_max, 5), 1))  # 6 evenly spaced X ticks
        ax.set_yticks(np.round(np.linspace(0, y_max, 6), 1))  # 6 evenly spaced Y ticks
        ax.tick_params(axis='both', labelsize=16)

    # Adjust layout
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.15)

    # Add titles under each column
    fig.text(0.18, 0.025, '(a) Self-Distillation', ha='center', va='center', fontsize=25)
    fig.text(0.52, 0.025, '(b) Ensemble', ha='center', va='center', fontsize=25)
    fig.text(0.84, 0.025, '(c) MCDropout', ha='center', va='center', fontsize=25)

    try:
        os.makedirs(pjoin(cwd, 'data', 'figs'))
    except FileExistsError:
        # directory already exists
        pass

    plt.savefig(pjoin(cwd, 'data', 'figs', f'mimic_entropy_density_self_vs_ensemble_aggregated_vs_mcd.pdf'))
    plt.close()

    # Create the plot in Gray-Scale
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Row 1, Column 1 - Self-Distillation (Blue)
    sns.kdeplot(self_dist_entropy_test, color='black', ax=axes[0, 0])
    axes[0, 0].set_title("ID Data", fontsize=21)
    axes[0, 0].set_xlabel("Entropy Values", fontsize=21)
    axes[0, 0].set_ylabel("Density", fontsize=21)

    # Row 1, Column 1 - MCDropout (Blue)
    sns.kdeplot(mc_dropout_entropy_test, color='black', ax=axes[0, 2])
    axes[0, 2].set_title("ID Data", fontsize=21)
    axes[0, 2].set_xlabel("Entropy Values", fontsize=21)
    axes[0, 2].set_ylabel("Density", fontsize=21)

    # Row 1, Column 2 - Ensemble (Blue)
    sns.kdeplot(ens_entropy_test, color='black', ax=axes[0, 1])
    axes[0, 1].set_title("ID Data", fontsize=21)
    axes[0, 1].set_xlabel("Entropy Values", fontsize=21)
    axes[0, 1].set_ylabel("Density", fontsize=21)

    # Row 2, Column 1 - Self-Distillation (Red)
    sns.kdeplot(self_dist_entropy_ood, color='dimgray', ax=axes[1, 0])
    axes[1, 0].set_title("OOD Data", fontsize=21)
    axes[1, 0].set_xlabel("Entropy Values", fontsize=21)
    axes[1, 0].set_ylabel("Density", fontsize=21)

    # Row 2, Column 1 - MCDRopout (Red)
    sns.kdeplot(mc_dropout_entropy_ood, color='dimgray', ax=axes[1, 2])
    axes[1, 2].set_title("OOD Data", fontsize=21)
    axes[1, 2].set_xlabel("Entropy Values", fontsize=21)
    axes[1, 2].set_ylabel("Density", fontsize=21)

    # Row 2, Column 2 - Ensemble (Red)
    sns.kdeplot(ens_entropy_ood, color='dimgray', ax=axes[1, 1])
    axes[1, 1].set_title("OOD Data", fontsize=21)
    axes[1, 1].set_xlabel("Entropy Values", fontsize=21)
    axes[1, 1].set_ylabel("Density", fontsize=21)

    # Set main titles for columns
    axes[0, 0].set_ylabel("Density", fontsize=21)
    axes[1, 0].set_ylabel("Density", fontsize=21)

    for ax in axes.flatten():
        # Dynamically set 6 ticks using np.linspace()
        _, x_max = ax.get_xlim()  # Get X-axis range
        _, y_max = ax.get_ylim()  # Get Y-axis range

        ax.set_xticks(np.round(np.linspace(0, x_max, 5), 1))  # 6 evenly spaced X ticks
        ax.set_yticks(np.round(np.linspace(0, y_max, 6), 1))  # 6 evenly spaced Y ticks
        ax.tick_params(axis='both', labelsize=16)

    # Adjust layout
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.15)

    # Add titles under each column
    fig.text(0.18, 0.025, '(a) Self-Distillation', ha='center', va='center', fontsize=25)
    fig.text(0.52, 0.025, '(b) Ensemble', ha='center', va='center', fontsize=25)
    fig.text(0.84, 0.025, '(c) MCDropout', ha='center', va='center', fontsize=25)

    try:
        os.makedirs(pjoin(cwd, 'data', 'figs'))
    except FileExistsError:
        # directory already exists
        pass

    plt.savefig(pjoin(cwd, 'data', 'figs', f'mimic_entropy_density_self_vs_ensemble_aggregated_vs_mcd-gray.pdf'))
