import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


def train(model, train_loader, device, criterion, optimizer):
    model.train()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        iteration_count += 1
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(out.cpu().detach().numpy())  # Logit values.
        loss_all += loss.detach().item()

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, true_classes, predicted_classes, logit_vals, loss_val, optimizer


def test(model, loader, device, criterion):
    model.eval()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        iteration_count += 1
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)

        pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(out.cpu().detach().numpy())  # Logit values.
        loss = criterion(out, data.y)
        loss_all += loss.item() * data.num_graphs

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, true_classes, predicted_classes, logit_vals, loss_val


def train_heterogeneous(model, train_loader, device, criterion, optimizer):
    model.train()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        iteration_count += 1
        data = data.to(device)
        batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
        out = model(data.x_dict, data.edge_index_dict, batch_dict)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(out.cpu().detach().numpy())  # Logit values.
        loss_all += loss.detach().item()

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, true_classes, predicted_classes, logit_vals, loss_val, optimizer


def test_heterogeneous(model, loader, device, criterion):
    model.eval()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        iteration_count += 1
        data = data.to(device)
        batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
        out = model(data.x_dict, data.edge_index_dict, batch_dict)

        pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(out.cpu().detach().numpy())  # Logit values.
        loss = criterion(out, data.y)
        loss_all += loss.item() * data.num_graphs

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, true_classes, predicted_classes, logit_vals, loss_val


def train_self_distillation(model, train_loader, device, criterion, optimizer, distillation_loss_coefficients,
                            feature_penalty_coefficients):
    model.train()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        iteration_count += 1
        data = data.to(device)
        output_list, feature_list = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(data.y, output_list, output_list[-1], feature_list, feature_list[-1],
                         distillation_loss_coefficients, feature_penalty_coefficients)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        pred = output_list[-1].detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(output_list[-1].cpu().detach().numpy())  # Logit values.
        loss_all += loss.detach().item()

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, loss_val, optimizer


def test_self_distillation(model, loader, device, criterion, distillation_loss_coefficients,
                           feature_penalty_coefficients):
    model.eval()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        iteration_count += 1
        data = data.to(device)
        output_list, feature_list = model(data.x, data.edge_index, data.batch)

        pred = output_list[-1].detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(output_list[-1].cpu().detach().numpy())  # Logit values.
        loss = criterion(data.y, output_list, output_list[-1], feature_list, feature_list[-1],
                         distillation_loss_coefficients, feature_penalty_coefficients)
        loss_all += loss.item() * data.num_graphs

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, loss_val


def train_self_distillation_heterogeneous(model, train_loader, device, criterion, optimizer,
                                          distillation_loss_coefficients, feature_penalty_coefficients):
    model.train()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        iteration_count += 1
        data = data.to(device)
        batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
        output_list, feature_list = model(data.x_dict, data.edge_index_dict,
                                          batch_dict)  # Perform a single forward pass.
        loss = criterion(data.y, output_list, output_list[-1], feature_list, feature_list[-1],
                         distillation_loss_coefficients, feature_penalty_coefficients)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        pred = output_list[-1].detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(output_list[-1].cpu().detach().numpy())  # Logit values.
        loss_all += loss.detach().item()

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, loss_val, optimizer


def test_self_distillation_heterogeneous(model, loader, device, criterion, distillation_loss_coefficients,
                                         feature_penalty_coefficients):
    model.eval()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        iteration_count += 1
        data = data.to(device)
        batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
        output_list, feature_list = model(data.x_dict, data.edge_index_dict,
                                          batch_dict)

        pred = output_list[-1].detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(output_list[-1].cpu().detach().numpy())  # Logit values.
        loss = criterion(data.y, output_list, output_list[-1], feature_list, feature_list[-1],
                         distillation_loss_coefficients, feature_penalty_coefficients)
        loss_all += loss.item() * data.num_graphs

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, loss_val


# Monte Carlo inference with last layer DROPOUT active (SELECTIVE DROPOUT)
def mc_inference(model, data, n_samples=30):
    model.eval()  # Disable all dropouts by default

    # Enable only the FC dropout manually
    model.dropout_fc.train()

    outs = [model(data.x, data.edge_index, data.batch) for _ in range(n_samples)]

    return outs


# Monte Carlo inference with last layer DROPOUT active (SELECTIVE DROPOUT)
def mc_inference_heterogeneous(model, data, n_samples=30):
    model.eval()  # Disable all dropouts by default

    # Enable only the FC dropout manually
    model.dropout_fc.train()

    batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
    outs = [model(data.x_dict, data.edge_index_dict, batch_dict) for _ in range(n_samples)]

    return outs
