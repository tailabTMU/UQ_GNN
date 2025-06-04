def find_upper_bound(no_layers: int, weight_func_type: str = 'nonlinear'):
    from .metrics import nonlinear_weight, weighted_agreement, \
        linear_weight  # To prevent a circular import
    import numpy as np
    import math

    if no_layers < 2:
        raise ValueError("Number of layers (including the reference) cannot be less than 2")
    if weight_func_type not in ['nonlinear', 'linear']:
        raise ValueError("Weight function type must be either 'nonlinear' or 'linear'")
    layers_weight = []
    layers_div = []
    for i in range(1, no_layers):
        if weight_func_type == 'nonlinear':
            layers_weight.append(nonlinear_weight(no_layers, no_layers - i, [0], [1]))
        else:
            layers_weight.append(linear_weight(no_layers, no_layers - i, [0], [1]))
        # Using math.pow and np.sqrt to have the same precision as the jensenshannon method in scipy
        layers_div.append([math.pow(np.sqrt(math.log(2, math.e)), 2)])

    test_agreement_values = weighted_agreement(layers_weight, layers_div)
    return test_agreement_values[0]
