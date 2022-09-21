import torch
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm

# def row_reduction(Y):
#     return np.mean(Y, axis=1) - 5*np.std(Y, axis=1) * np.sign(np.mean(Y[:,:10], axis=1) -
#                                             np.mean(Y[:, 11:], axis=1))

def row_reduction(Y):
    return np.mean(Y, axis=1, keepdims=True) + len(Y[0]) * np.min(Y, axis=1, keepdims=True)

def optimize_gp(gmodel):
    best_result = np.inf  # Arbitrary at this point
    best_results = []
    hp = None
    lr1_list = np.logspace(-2, -1, 30)
    lr2_list = np.logspace(-3, -2, 30)
    gamma_list = np.arange(0.1, 0.5, 0.05)
    decay_list = np.logspace(-4, -3, 30)
    n_combis = len(lr1_list)*len(lr2_list)*len(gamma_list)*len(decay_list)
    print(f'Trying {n_combis} Combinations.')
    all_combinations = itertools.product(lr1_list, lr2_list, gamma_list, decay_list)
    for comb in tqdm(all_combinations, total=n_combis):
        X = np.array(comb).reshape(1, -1)
        result = gmodel.predict(X)
        if result < best_result:
            best_result = result
            hp = X
            best_results.append(best_result)
    return hp, best_result, best_results


file_path = "checkpoints/GP/test_losses.pth"
TS = torch.load(file_path)

X = np.array(TS['params'])
Y = np.array(TS['test_losses'])

Y = row_reduction(Y)

gmodel = GaussianProcessRegressor()
gmodel.fit(X, Y)

hp, best, all = optimize_gp(gmodel=gmodel)
print("")