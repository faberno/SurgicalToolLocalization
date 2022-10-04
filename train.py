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
"""Performs training of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
def train(config_file, export=False):
    print('Reading config file...')
    configuration = parse_configuration(config_file)
    configuration['checkpoint_folder'] = os.path.join(configuration['model_params']['checkpoint_path'],
                                                      datetime.now().strftime("%d_%m_%Y-%H:%M"))
    try:
        os.makedirs(configuration['checkpoint_folder'])
    except OSError as exc:
        print("Could not create checkpoint folder")

    print('Initializing dataset...')
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
    model = create_model(configuration['model_params'])
    model = model.to(model.device)
    starting_epoch = model.setup()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params'])   # create a visualizer that displays images and plots

    print('Initializing AP_Tester...')
    ap_tester = AP_tester(val_dataset.dataset, model.device, val_dataset.dataset.resize,
                          model.configuration['backbone']['options']['strides'],
                          model.configuration['backbone']['name'])

    num_epochs = configuration['model_params']['max_epochs']
    for epoch in range(starting_epoch + 1, num_epochs + 1):
        epoch_start_time = time.time()  # timer for entire epoch

        train_iterations = len(train_dataset)
        train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']

        model.train()
        model.train_batch_losses = []

        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            model.train_minibatch(data)
            if i % configuration['printout_freq'] == 0:
                visualizer.print_current_train_loss(epoch, num_epochs, i, math.floor(train_iterations / train_batch_size), model.train_batch_losses[-1])
        model.train_losses.append(torch.mean(torch.tensor(model.train_batch_losses)).item())

        model.eval()
        model.test_batch_losses = []

        print("Validating...")
        for i, data in tqdm(enumerate(val_dataset)):
            model.test_minibatch(data, ap_tester)
        AP_loc = False
        if train_dataset.dataset.inference and ((epoch % configuration['AP_loc_freq'] == 0) or (epoch == num_epochs)):
            AP_loc = True
        AP = ap_tester.run(compute_AP_loc=AP_loc)

        model.test_losses.append((torch.sum(torch.tensor(model.test_batch_losses)) / len(val_dataset)).item())
        model.det_APs.append(AP['det_AP'])
        model.det_APs_cw.append(AP['det_AP_cw'])
        if AP_loc:
            model.loc_APs.append(AP['loc_AP'])
            model.loc_APs_cw.append(AP['loc_AP_cw'])
        visualizer.print_current_epoch_loss(epoch, num_epochs, model=model, plot=True, AP=AP)

        best = False
        if (epoch > 1) and (model.test_losses[-1] < min(model.test_losses[:-1])):
            best = True

        print(f'Saving model at the end of epoch {epoch}')
        model.save_network(epoch, configuration['checkpoint_folder'], best, AP)

        print(f'End of epoch {epoch} / {num_epochs} \t Time Taken: {time.time() - epoch_start_time} sec')

        model.scheduler.step()

if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)