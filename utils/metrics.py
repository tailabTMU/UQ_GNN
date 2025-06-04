import math

import numpy as np
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
import warnings


def jensenshannon_metric(p: list[float], q: list[float]) -> float:
    # p is the True probability distribution
    # q is a model's probability distribution
    if len(p) != len(q):
        raise Exception('Lengths of two arrays must match')
    return math.pow(jensenshannon(p, q, base=math.e), 2)


def kl_divergence(p: list[float], q: list[float]) -> float:
    # p is the True probability distribution
    # q is a model's probability distribution
    if len(p) != len(q):
        raise Exception('Lengths of two arrays must match')
    return sum(rel_entr(p[i], q[i]) for i in range(len(p)))


def agreement(values: list[list[float]]) -> list[float]:
    return np.array(values).sum(axis=0).tolist()


# def weighted_agreement(weights: list[list[float]], values: list[list[float]]) -> list[float]:
#     if np.array(weights).shape != np.array(values).shape:
#         raise Exception('Shape of the weight and the values arrays must match')
#     weighted_kl_values = np.multiply(np.array(weights), np.array(values))
#     return weighted_kl_values.sum(axis=0).tolist()

def weighted_agreement(weights: list[list[float]], values: list[list[float]], normalize: bool = False,
                       num_layers: int = None, weight_func_type: str = None) -> list[float]:
    if np.array(weights).shape != np.array(values).shape:
        raise Exception('Shape of the weight and the values arrays must match')
    if normalize:
        warnings.warn('You should only use the normalized version with Jensen-Shannon Divergence')
        from utils.extras import find_upper_bound  # To prevent a circular import
        if num_layers is None:
            raise Exception('Normalization requires a number of layers')
        elif num_layers < 2:
            raise Exception('Normalization requires at least 2 layers')
        if weight_func_type is None:
            raise Exception('Normalization requires the weight function type')
        elif weight_func_type not in ['nonlinear', 'linear']:
            raise ValueError("Weight function type must be either 'nonlinear' or 'linear'")
        normalization_param = find_upper_bound(num_layers, weight_func_type=weight_func_type)
    else:
        normalization_param = 1

    weighted_div_values = np.multiply(np.array(weights), np.array(values))
    return (weighted_div_values.sum(axis=0) / normalization_param).tolist()


def linear_weight(num_layers: int, distance_from_ref: int, classes: list[int], ref_classes: list[int]) -> list[float]:
    if distance_from_ref < 0 or distance_from_ref >= num_layers:
        raise Exception('Distance from ref must be between 0 and num_layers')
    if len(classes) != len(ref_classes):
        raise Exception('Lengths of predicted classes and reference classes must match')
    indicator = (np.array(classes) != np.array(ref_classes)).astype(int)
    return ((indicator * ((num_layers - distance_from_ref) / num_layers)) + 1).tolist()


def nonlinear_weight_2(num_layers: int, distance_from_ref: int, classes: list[int], ref_classes: list[int],
                       delta: float = None) -> list[float]:
    if delta is None:
        delta = 1 / num_layers
    if delta <= 0 and delta >= 1:
        raise Exception('Delta must be between 0 and 1')
    if distance_from_ref < 0 or distance_from_ref >= num_layers:
        raise Exception('Distance from ref must be between 0 and num_layers')
    if len(classes) != len(ref_classes):
        raise Exception('Lengths of predicted classes and reference classes must match')
    indicator = (np.array(classes) != np.array(ref_classes)).astype(int)
    weight = (math.log2(num_layers - distance_from_ref + delta) / math.log2(num_layers + delta)) + 1
    return [math.pow(weight, i) for i in indicator]


def nonlinear_weight(num_layers: int, distance_from_ref: int, classes: list[int], ref_classes: list[int]) -> list[
    float]:
    if distance_from_ref < 0 or distance_from_ref >= num_layers:
        raise Exception('Distance from ref must be between 0 and num_layers')
    if len(classes) != len(ref_classes):
        raise Exception('Lengths of predicted classes and reference classes must match')

    power = distance_from_ref - num_layers
    indicator = (np.array(classes) != np.array(ref_classes)).astype(int)
    weight = (math.pow(math.e, power) * -1) + 2
    return [math.pow(weight, i) for i in indicator]
