# GNN's Uncertainty Quantification using Self-Distillation

This code is the implementation of quantifying uncertainty of GNNs using Self-Distillation proposed in "GNN's Uncertainty Quantification using Self-Distillation" submitted to the International Conference on AI in Healthcare (AIiH) 2025. The proposed method has been evaluated on a graph classification task, but it can be extended to other tasks as well.

The publicly available [Enzymes Dataset collected by TU Dortmund University](https://chrsmrrs.github.io/datasets/) and [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) have been used for evaluations.

<strong>Paper:</strong> Hirad Daneshvar and Reza Samavi. "GNN's Uncertainty Quantification using Self-Distillation." International Conference on AI in Healthcare, 2025.

## Setup
The code has been dockerized (using GPU). The requirements are included in the requirements.txt file. If you choose to use docker, you don't need to install the packages as it will automatically install them all. To use the docker, make sure you create a copy of _.env.example_ file and name it _.env_ and complete it according to your system. To use the dockerized version, you will need a Ubuntu based system.

If you choose to run the code using CPU, you don't need to use docker as the requirements for CPU support is included in a file called _requirements_cpu.txt_.

## Running Experiments
After creating the _.env_ file, you first need to build the image using ```docker compose build```. Then you need to run ```docker compose up -d``` to start the project. To run the experiments, you need to run the following:
### Enzymes Dataset
- Training an ensemble of GNNs separately:
  - ```docker compose exec torch bash -c "python3.9 enzymes_graph_classification_ensemble.py"```
- Training the MC Dropout model:
  - ```docker compose exec torch bash -c "python3.9 enzymes_graph_classification_mc_dropout.py"```
- Training multiple GNN classifiers using Self-Distillation:
  - ```docker compose exec torch bash -c "python3.9 enzymes_graph_classification_self_distillation.py"```
> Note: You don't need to download the datasets. The datasets will be automatically downloaded.
### MIMIV-IV Dataset
- Download the `admissions.csv`, `icustays.csv`, and `sevice.csv` files from the source.
- Create a `data` directory if it does not exist. Inside `data` create a directory called `MIMICDataset`. Inside `MIMICDataset` create a directory called `raw`. Copy all the downloaded `csv` files in this directory.
- Training an ensemble of GNNs separately (although early stopping is implemented, the model trains for the full epoch to be able to compare training times):
  - ```docker compose exec torch bash -c "python3.9 mimic_graph_classification_ensemble.py"```
- Training the MC Dropout model (although early stopping is implemented, the model trains for the full epoch to be able to compare training times):
  - ```docker compose exec torch bash -c "python3.9 mimic_graph_classification_mc_dropout.py"```
- Training multiple GNN classifiers using Self-Distillation (although early stopping is implemented, the model trains for the full epoch to be able to compare training times):
  - ```docker compose exec torch bash -c "python3.9 mimic_graph_classification_self_distillation.py"```

## Generating Results
After running the experiments, the model outputs and metadata will be saved in:
### Enzymes Dataset
- A folder called `saved_info_graph_classification` in the root directory of the project for graph classification
  - For the ensemble there would be a folder called `ensemble_enzymes` in the `saved_info_graph_classification` director
  - For the MC Dropout there would be a folder called `mc_dropout_enzymes` in the `saved_info_graph_classification` director
  - For the Self-Distillation approach there would be a folder called `self_distillation_enzymes` in the `saved_info_graph_classification` director

You need to run the following commands to see the results:
- ```docker compose exec torch bash -c "python3.9 enzymes_graph_classification_metrics.py"```: Prints a table with performance and time to train comparisons of the Ensemble and the Self-Distillation approaches.
- ```docker compose exec torch bash -c "python3.9 plot_enzymes_graph_classification_entropy.py"```: Creates the KDE plots for entropy of the MC Dropout, Ensemble and the Self-Distillation approaches on both OOD and ID data. The plot is stored both in color and in gray-scale inside the `data/figs` directory as PDF files.
- ```docker compose exec torch bash -c "python3.9 plot_enzymes_graph_classification_uncertainty_kd_plot.py"```: Creates the KDE plot for the weighted disagreement values on a subset of the ID data using the Self-Distillation approach. The plot is stored both in color and in gray-scale inside the `data/figs` directory as PDF files.
- ```docker compose exec torch bash -c "python3.9 plot_enzymes_graph_classification_uncertainty_jsd_plot.py"```: Creates the KDE plot for the proposed metric on the ID data using the Self-Distillation approach. The plot is stored both in color and in gray-scale inside the `data/figs` directory as PDF files.
### MIMIC Dataset
- A folder called `saved_info_graph_classification` in the root directory of the project for graph classification
  - For the ensemble there would be a folder called `ensemble_mimic` in the `saved_info_graph_classification` director
  - For the MC Dropout there would be a folder called `mc_dropout_mimic` in the `saved_info_graph_classification` director
  - For the Self-Distillation approach there would be a folder called `self_distillation_mimic` in the `saved_info_graph_classification` director

You need to run the following commands to see the results:
- ```docker compose exec torch bash -c "python3.9 mimic_graph_classification_metrics.py"```: Prints a table with performance and time to train comparisons of the Ensemble and the Self-Distillation approaches.
- ```docker compose exec torch bash -c "python3.9 plot_mimic_graph_classification_entropy.py"```: Creates the KDE plots for entropy of the MC Dropout, Ensemble and the Self-Distillation approaches on both OOD and ID data. The plot is stored both in color and in gray-scale inside the `data/figs` directory as PDF files.
- ```docker compose exec torch bash -c "python3.9 plot_mimic_graph_classification_uncertainty_kd_plot.py"```: Creates the KDE plot for the weighted disagreement values on a subset of the ID data using the Self-Distillation approach. The plot is stored both in color and in gray-scale inside the `data/figs` directory as PDF files.
- ```docker compose exec torch bash -c "python3.9 plot_mimic_graph_classification_uncertainty_jsd_plot.py"```: Creates the KDE plot for the proposed metric on the ID data using the Self-Distillation approach. The plot is stored both in color and in gray-scale inside the `data/figs` directory as PDF files.

## Cite
TBD