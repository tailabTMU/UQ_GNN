import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from models import SelfDistillationSAGE as SAGE
from utils.gnn import train_self_distillation as train, test_self_distillation as test
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
from tqdm import tqdm
import random
import glob
import time
from sklearn.metrics import f1_score, accuracy_score
from utils.loss import SelfDistillationLoss

def main(args):
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    configs = {
        "params": {
            # "epochs": 1000,
            "epochs": 200,
            "batch_size": 20,
            "init_lr": 7e-4,
            "lr_reduce_factor": 0.5,
            "lr_schedule_patience": 25,
            "min_lr": 1e-6,
            "weight_decay": 0.0,
            "print_epoch_interval": 5,
        },
        'net_params': {
            'L': 4,
            'hidden_dim': 90,
            'out_dim': 90,
            'residual': True,
            'readout': 'mean',
            'in_feat_dropout': 0.0,
            'dropout': 0.0,
            'batch_norm': True,
            'sage_aggregator': 'max'
        }
    }

    config = configs['net_params']
    params = configs['params']

    try:
        os.makedirs(pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes'))
    except FileExistsError:
        # directory already exists
        pass

    dataset = TUDataset(root=os.path.join(os.getcwd(), 'data', 'TUDataset'), name='enzymes'.upper(), use_node_attr=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Device: {device}')

    ood_target = 5
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

    iid_data = np.array(iid_data_idx)
    labels = np.array(iid_data_y)

    ood_dataset = dataset.index_select(ood_data_idx)
    ood_loader = DataLoader(ood_dataset, batch_size=params['batch_size'], shuffle=False,
                            collate_fn=ood_dataset.collate)

    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    for alpha_val in [0.1, 0.4, 0.6, 0.8]:
        split_number = 0
        test_predicted_classes = []
        test_true_classes = []
        avg_train_time = []
        print(f'Alpha Val: {alpha_val}')
        for train_index, test_index in skf.split(iid_data, labels):
            split_number += 1
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(1)
            train_dataset_idx, test_dataset_idx = iid_data[train_index], iid_data[test_index]
            train_y, test_y = labels[train_index], labels[test_index]
            train_dataset_idx, validation_dataset_idx, train_y, val_y = train_test_split(train_dataset_idx, train_y,
                                                                                         test_size=0.2,
                                                                                         random_state=1,
                                                                                         stratify=train_y)

            train_dataset = dataset.index_select(train_dataset_idx)
            validation_dataset = dataset.index_select(validation_dataset_idx)
            test_dataset = dataset.index_select(test_dataset_idx)

            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                      collate_fn=train_dataset.collate)
            val_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False,
                                    collate_fn=validation_dataset.collate)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                     collate_fn=test_dataset.collate)

            # Begin training
            start_time = time.time()

            model_name = 'sage'
            model = SAGE(dataset.num_features, num_classes, config)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=params['lr_reduce_factor'],
                                                                   patience=params['lr_schedule_patience'],
                                                                   verbose=False)
            criterion = SelfDistillationLoss()
            distillation_loss_coefficients = [alpha_val for _ in range(config['L'] - 1)]
            distillation_loss_coefficients.append(0)
            feature_penalty_coefficients = [0.04 for _ in range(config['L'] - 1)]
            feature_penalty_coefficients.append(0)

            root_ckpt_dir = pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes')

            print(f'Training Model for Split {split_number}')
            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    if epoch >= 180:
                        distillation_loss_coefficients = [0 for _ in range(config['L'])]
                        feature_penalty_coefficients = [0 for _ in range(config['L'])]
                    t.set_description('Epoch %d' % epoch)

                    epoch_train_acc, epoch_train_loss, optimizer = train(model, train_loader, device, criterion,
                                                                         optimizer, distillation_loss_coefficients,
                                                                         feature_penalty_coefficients)
                    epoch_val_acc, epoch_val_loss = test(model, val_loader, device, criterion,
                                                         distillation_loss_coefficients,
                                                         feature_penalty_coefficients)

                    t.set_postfix(lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  )

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, f"Self_Distillation_RUN_" + str(split_number))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(model.state_dict(), '{}.pkl'.format(pjoin(ckpt_dir, f"epoch_{str(epoch)}")))

                    files = glob.glob(pjoin(ckpt_dir, '*.pkl'))
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch - 1:
                            os.remove(file)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]['lr'] < params['min_lr']:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break

            train_time = (time.time() - start_time)

            logit_vals = []
            predicted_classes = []
            for layer in range(config['L']):
                logit_vals.append([])
                predicted_classes.append([])
            true_classes = []
            for data in test_loader:
                data = data.to(device)
                true_classes += list(data.y.cpu().numpy())
                output_list, _ = model(data.x, data.edge_index, data.batch)
                for idx, out in enumerate(output_list):
                    pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
                    logit_vals[idx] += list(out.cpu().detach().numpy())  # Logit values.
                    predicted_classes[idx] += list(pred.cpu().numpy())

            # print("Test Data", classification_report(true_classes, predicted_classes[-1]))

            for layer in range(config['L']):
                with open(pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes',
                                f'{model_name}_layer_{(layer + 1)}_split_{split_number}_alpha_{alpha_val}test_logit_vals.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile, logit_vals[layer])

                with open(
                        pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes',
                              f'{model_name}_layer_{(layer + 1)}_split_{split_number}_alpha_{alpha_val}test_true_classes.txt'),
                        'w+') as myfile:
                    np.savetxt(myfile, true_classes)

                with open(pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes',
                                f'{model_name}_layer_{(layer + 1)}_split_{split_number}_alpha_{alpha_val}test_predicted_classes.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile, predicted_classes[layer])


            with open(pjoin(cwd, 'saved_info_graph_classification_ablation', 'self_distillation_enzymes',
                            f'{model_name}_split_{split_number}_alpha_{alpha_val}time_to_train.txt'),
                      'w+') as myfile:
                myfile.write(f'Time to Train (s): {train_time}')

            test_true_classes.append(true_classes)
            test_predicted_classes.append(predicted_classes[-1])
            avg_train_time.append(train_time)


        accuracies = []
        f1_scores = []
        for idx, split_prediction in enumerate(test_predicted_classes):
            accuracies.append(accuracy_score(test_true_classes[idx], split_prediction))
            f1_scores.append(f1_score(test_true_classes[idx], split_prediction, average='weighted'))
        print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.t.d. {:.4f}""".format(
            np.mean(np.array(accuracies)) * 100, np.std(accuracies) * 100))
        print("""\n\n\nFINAL RESULTS\n\nTEST F1 Score averaged: {:.4f} with s.t.d. {:.4f}""".format(
            np.mean(np.array(f1_scores)) * 100, np.std(f1_scores) * 100))

        print("""\n\n\nFINAL RESULTS\n\nTRAINING Time (s) averaged: {:.4f} with s.t.d. {:.4f}""".format(
            np.mean(avg_train_time), np.std(avg_train_time)))
        print()
        print('=============================================================================')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an Ensemble of Graph Classifiers using Self-Distillation')
    main(parser.parse_args())
