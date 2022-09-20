import argparse
from datasets import create_dataset
from utils import parse_configuration
import math
from models import create_model
import time
from utils.visualizer import Visualizer
import torch
from datetime import datetime
import os
from utils.AP_tester import AP_tester
from tqdm import tqdm
import multiprocessing
import itertools

"""Trains models on all combinations of Hyperparameters and saves their validation loss."""

hyperparameters = {
    'lr1': [0.01, 0.1],
    'lr2': [0.001, 0.01],
    'gamma': [0.1, 0.5],
    'decay': [1e-4, 1e-3]
}

epochs = 50
lr_milestones = [25, 40, 50]
batch_size = 50

def train(config_file):
    print('Reading config file...')
    configuration = parse_configuration(config_file)
    configuration['checkpoint_folder'] = os.path.join(configuration['model_params']['checkpoint_path'],
                                                      "GP")
    try:
        os.makedirs(configuration['checkpoint_folder'])
    except OSError as exc:
        print("Could not create checkpoint folder")

    print('Initializing dataset...')
    configuration['train_dataset_params']['loader_params']['batch_size'] = batch_size
    configuration['val_dataset_params']['loader_params']['batch_size'] = batch_size
    train_dataset = create_dataset(configuration['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    configuration['model_params']['n_classes'] = train_dataset.dataset.n_classes
    configuration['model_params']['classes'] = train_dataset.dataset.classes
    configuration['model_params']['img_size'] = train_dataset.dataset.resize
    configuration['model_params']['train_weights'] = torch.tensor(train_dataset.dataset.weights)
    print(f'The number of training samples = {train_dataset_size}')

    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print(f'The number of validation samples = {val_dataset_size}')

    print('Initializing model...')
    configuration['model_params']['max_epochs'] = epochs


    HP_keys, HP_values = zip(*hyperparameters.items())
    all_combinations = list(itertools.product(*HP_values))
    all_test_losses = []
    for c, comb in enumerate(all_combinations):
        print(f"Combination {c}: {comb}")
        params = dict(zip(HP_keys, comb))
        configuration['model_params']['optimizer']['lr'] = [params['lr1'], params['lr2']]
        configuration['model_params']['optimizer']['lr_milestones'] = lr_milestones
        configuration['model_params']['optimizer']['gamma'] = params['gamma']
        configuration['model_params']['optimizer']['weight_decay'] = params['decay']
        model = create_model(configuration['model_params'])
        model = model.to(model.device)


        num_epochs = configuration['model_params']['max_epochs']
        for epoch in tqdm(range(1, num_epochs + 1)):
            model.train()
            model.train_batch_losses = []

            for i, data in enumerate(train_dataset):  # inner loop within one epoch
                model.train_minibatch(data)

            model.eval()
            for i, data in enumerate(val_dataset):
                model.test_minibatch(data)

            model.test_losses.append(
                (torch.sum(torch.tensor(model.test_batch_losses)) / len(val_dataset)).item())

            model.scheduler.step()
            model.test_batch_losses = []

        all_test_losses.append(model.test_losses)

        print(f'Saving model...')
        filename = f"{c}_lr1{params['lr1']}_lr2{params['lr2']}_gamma{params['gamma']}_decay{params['decay']}.pth"
        save_path = os.path.join(configuration['checkpoint_folder'],  filename)

        if model.use_cuda:
            model_state = model.cpu().state_dict()
            model.to(model.device)
        else:
            model_state = model.cpu().state_dict()

        params['model_state_dict'] = model_state
        params['test_losses'] = model.test_losses
        torch.save(params, save_path)
        torch.save({'params': all_combinations[: (c+1)],
                    'test_losses': all_test_losses},
                   os.path.join(configuration['checkpoint_folder'],  "test_losses.pth"))

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)