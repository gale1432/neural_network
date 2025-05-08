import numpy as np
from collections import defaultdict

def calculate_stats(arr):
    minimum = np.min(arr)
    maximum = np.max(arr)
    sd = np.std(arr)
    var = np.var(arr)
    mean = np.mean(arr)
    return [minimum, maximum, sd, var, mean]

def get_stats_one_layer(weights, iter_number, layer_nodes, features):
    if layer_nodes == 20:
        layer_number = 1
    elif layer_nodes == 30:
        layer_number = 2
    else:
        layer_number = 3
    weight_stats = dict()
    db_arr = list()
    w_arr = list()
    dw_arr = list()
    b_arr = list()
    for i in range(features):
        for j in range(layer_nodes):
            for k in range(1, iter_number+1):
                w_arr.append(weights[str(k)][f'W{layer_number}'][j][i])
                dw_arr.append(weights[str(k)][f'dW{layer_number}'][j][i])
                #weight_stats.update({'W1'})
    weight_stats[f'W{layer_number}'] = calculate_stats(w_arr)
    weight_stats[f'dW{layer_number}'] = calculate_stats(w_arr)

    for i in range(layer_nodes):
        for j in range(1, iter_number+1):
            b_arr.append(weights[str(j)][f'b{layer_number}'][i])
    weight_stats[f'b{layer_number}'] = calculate_stats(b_arr)

    for i in range(1, iter_number+1):
        db_arr.append(weights[str(i)][f'db{layer_number}'])
    weight_stats[f'db{layer_number}'] = calculate_stats(db_arr)
    return weight_stats

def get_all_layer_stats(weights, iter_number):
    final_weight_stats = defaultdict(dict)
    final_weight_stats['l1'] = get_stats_one_layer(weights, iter_number, 20, 500)
    final_weight_stats['l2'] = get_stats_one_layer(weights, iter_number, 30, 20)
    final_weight_stats['l3'] = get_stats_one_layer(weights, iter_number, 11, 30)
    return final_weight_stats