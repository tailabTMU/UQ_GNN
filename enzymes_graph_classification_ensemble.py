import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from models import SAGE
from utils.gnn import train, test
from sklearn.metrics import classification_report
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
from tqdm import tqdm
import random
import glob
import time
from sklearn.metrics import f1_score, accuracy_score


# Function to compute majority vote for each sample
def majority_vote(predictions):
    # Unique values and their counts
    unique, counts = np.unique(predictions, return_counts=True)
    # Return the value with the highest count
    return unique[np.argmax(counts)]


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
        os.makedirs(pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes'))
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

    split_number = 0
    test_predicted_classes = {}
    test_true_classes = {}
    avg_train_time = {}
    for train_index, test_index in skf.split(iid_data, labels):
        split_number += 1
        test_predicted_classes[split_number] = []
        avg_train_time[split_number] = []
        test_true_classes[split_number] = []
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
        for ensemble_num in range(1, 5):
            start_time = time.time()
            # setting seeds
            random.seed(1 + ensemble_num)
            np.random.seed(1 + ensemble_num)
            torch.manual_seed(1 + ensemble_num)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(1 + ensemble_num)

            model_name = 'sage'
            model = SAGE(dataset.num_features, num_classes, config)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=params['lr_reduce_factor'],
                                                                   patience=params['lr_schedule_patience'],
                                                                   verbose=False)
            criterion = torch.nn.CrossEntropyLoss()

            root_ckpt_dir = pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes')

            print(f'Training Model for Split {split_number} Ensemble {ensemble_num}')
            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)

                    epoch_train_acc, _, _, _, epoch_train_loss, optimizer = train(model, train_loader, device,
                                                                                  criterion,
                                                                                  optimizer)
                    epoch_val_acc, _, _, _, epoch_val_loss = test(model, val_loader, device, criterion)

                    t.set_postfix(lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  )

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, f"Ensemble_{ensemble_num}_RUN_" + str(split_number))
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
            true_classes = []
            predicted_classes = []
            for data in test_loader:
                data = data.to(device)
                true_classes += list(data.y.cpu().numpy())
                out = model(data.x, data.edge_index, data.batch)
                pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
                logit_vals += list(out.cpu().detach().numpy())  # Logit values.
                predicted_classes += list(pred.cpu().numpy())

            print("Test Data", classification_report(true_classes, predicted_classes))

            with open(pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes',
                            f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_test_logit_vals.txt'),
                      'w+') as myfile:
                np.savetxt(myfile, logit_vals)

            with open(
                    pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes',
                          f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_test_true_classes.txt'),
                    'w+') as myfile:
                np.savetxt(myfile, true_classes)

            with open(pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes',
                            f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_test_predicted_classes.txt'),
                      'w+') as myfile:
                np.savetxt(myfile, predicted_classes)

            with open(pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes',
                            f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_time_to_train.txt'),
                      'w+') as myfile:
                myfile.write(f'Time to Train (s): {train_time}')

            test_predicted_classes[split_number].append(predicted_classes)
            test_true_classes[split_number] = true_classes
            avg_train_time[split_number].append(train_time)

            logit_vals = []
            predicted_classes = []
            for data in ood_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
                logit_vals += list(out.cpu().detach().numpy())  # Logit values.
                predicted_classes += list(pred.cpu().numpy())


            with open(pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes',
                            f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_ood_logit_vals.txt'),
                      'w+') as myfile:
                np.savetxt(myfile, logit_vals)

            with open(pjoin(cwd, 'saved_info_graph_classification', 'ensemble_enzymes',
                            f'{model_name}_ensemble_{ensemble_num}_split_{split_number}_ood_predicted_classes.txt'),
                      'w+') as myfile:
                np.savetxt(myfile, predicted_classes)

    accuracies = []
    f1_scores = []
    for split_number in range(1, (len(test_predicted_classes) + 1)):
        split_prediction = np.apply_along_axis(majority_vote, axis=1, arr=np.array(
            test_predicted_classes[split_number]).T)

        accuracies.append(accuracy_score(test_true_classes[split_number], split_prediction))
        f1_scores.append(f1_score(test_true_classes[split_number], split_prediction, average='weighted'))
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.t.d. {:.4f}""".format(
        np.mean(np.array(accuracies)) * 100, np.std(accuracies) * 100))
    print("""\n\n\nFINAL RESULTS\n\nTEST F1 Score averaged: {:.4f} with s.t.d. {:.4f}""".format(
        np.mean(np.array(f1_scores)) * 100, np.std(f1_scores) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an Ensemble of Graph Classifiers with CrossEntropy Loss')
    main(parser.parse_args())
