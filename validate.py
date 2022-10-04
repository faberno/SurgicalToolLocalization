import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
from utils.visualizer import Visualizer
from utils.AP_tester import AP_tester
import time
from tqdm import tqdm
from operator import itemgetter

"""Performs validation of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
"""
def validate(config_file):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    configuration['model_params']['n_classes'] = val_dataset.dataset.n_classes
    configuration['model_params']['classes'] = val_dataset.dataset.classes
    configuration['model_params']['img_size'] = val_dataset.dataset.resize
    print(f'The number of validation samples = {val_dataset_size}')

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model = model.to(model.device)
    model.setup()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params'])   # create a visualizer that displays images and plots

    print('Initializing AP_Tester...')
    ap_tester = AP_tester(val_dataset.dataset, model.device, val_dataset.dataset.resize, model.configuration['backbone']['options']['strides'],
                          model.configuration['backbone']['name'])

    start_time = time.time()  # timer for entire epoch

    model.eval()
    model.test_batch_losses = []

    print("Validating...")
    for i, data in enumerate(tqdm(val_dataset, total=len(val_dataset.dataloader))):
        output = model.test_minibatch(data, ap_tester)
        visualizer.plot_validation_images(data['img'], output['crm'], data['target'], output['peak_list'],
                                          itemgetter(*data['idx'])(ap_tester.all_bboxes))

    AP = ap_tester.run(compute_AP_loc=True)

    visualizer.print_current_epoch_loss(model=model, plot=True, AP=AP)

    print(f'Time Taken: {time.time() - start_time}')

if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    validate(args.configfile)