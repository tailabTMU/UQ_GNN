from torch.nn import functional as F
from torch import nn
import torch


class SelfDistillationLoss(nn.Module):
    def __init__(self):
        super(SelfDistillationLoss, self).__init__()

    def forward(self, true_labels, all_layers_predictions, reference_predictions, all_layer_features,
                reference_features, distillation_loss_coefficients, feature_penalty_coefficients):
        if not isinstance(distillation_loss_coefficients, list):
            raise Exception('Distillation loss coefficients must be a list')
        if not isinstance(feature_penalty_coefficients, list):
            raise Exception('Feature penalty coefficients must be a list')
        distillation_losses = []
        penalty_losses = []
        for layer_idx, layer_predictions in enumerate(all_layers_predictions):
            # 1. Calculate Distillation Loss for Each Layer
            ce_loss = F.cross_entropy(layer_predictions, true_labels, reduction='none').reshape([-1, 1])
            ce_loss = torch.mul(ce_loss, (1 - distillation_loss_coefficients[layer_idx]))
            kd_loss = torch.sum(F.kl_div(F.log_softmax(layer_predictions, 1), F.softmax(reference_predictions, dim=1),
                                         reduction='none'), dim=1).reshape([-1, 1])
            kd_loss = torch.mul(kd_loss, distillation_loss_coefficients[layer_idx])

            sum_loss = torch.add(ce_loss, kd_loss)
            distillation_losses.append(sum_loss)

            # 2. Calculate Shallow Feature Penalty for Each Layer
            pdist = nn.PairwiseDistance(p=2)
            layer_features = all_layer_features[layer_idx]
            penalty_loss = pdist(layer_features, reference_features) * feature_penalty_coefficients[layer_idx]
            penalty_losses.append(penalty_loss)

        distillation_loss = torch.mean(torch.stack(distillation_losses), dim=0)
        penalty_loss = torch.mean(torch.stack(distillation_losses), dim=0)

        loss = torch.add(distillation_loss, penalty_loss).mean()
        return loss
