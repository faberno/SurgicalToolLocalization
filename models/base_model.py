import os
import torch
from utils import get_scheduler
from utils import transfer_to_device
from collections import OrderedDict
from abc import ABC, abstractmethod
import torch.nn as nn
from datetime import datetime

class BaseModel(nn.Module):
    """This class is an abstract base class (ABC) for models.
    """

    def __init__(self, configuration):
        """Initialize the BaseModel class.

        Parameters:
            configuration: Configuration dictionary.

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define these lists:
            -- self.network_names (str list):       define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        super().__init__()
        self.configuration = configuration
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        torch.backends.cudnn.benchmark = True
        self.save_dir = configuration['checkpoint_path']
        if 'train_weights' in configuration:
            self.train_weights = configuration['train_weights'].to(self.device)

        self.train_batch_losses = []
        self.test_batch_losses = []

        self.train_losses = []
        self.test_losses = []

        self.det_APs = []
        self.det_APs_cw = []

        self.loc_APs = []
        self.loc_APs_cw = []

        self.classes = configuration['classes']

    def train_minibatch(self, inputs):
        data = transfer_to_device(inputs['img'], self.device)
        label = transfer_to_device(inputs['target'], self.device)

        output = self(data)
        loss = self.criterion_loss(output, label, weight=self.train_weights, reduction='mean')
        loss.backward()  # calculate gradients
        self.train_batch_losses.append(loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad()

    def test_minibatch(self, inputs, ap_tester=None):
        data = transfer_to_device(inputs['img'], self.device)
        label = transfer_to_device(inputs['target'], self.device)
        if len(data.shape) == 4:
            batch_size = len(data)
        else:
            batch_size = 1
        with torch.no_grad():
            output = self(data, return_crm=True)

            if 'peak_list' in output:
                new_peaks = []
                new_vals = []
                for k in range(batch_size):
                    mask = output['peak_list'][:, 0] == k
                    coords = output['peak_list'][mask, 1:].cpu()
                    vals = output['peak_values'][mask].cpu()
                    new_peaks.append(coords)
                    new_vals.append(vals)
                output['peak_list'] = new_peaks
                output['peak_values'] = new_vals

            if ap_tester is not None:
                ap_tester.update(output, inputs['idx'])
            loss = self.criterion_loss(output['class_scores'], label, reduction='sum')
            self.test_batch_losses.append(loss.item())
        return output

    def setup(self):
        """
        Restore full state of the network including scheduler, optimizer and statistics.
        """
        if self.configuration['load_checkpoint']:
            checkpoint_path = self.configuration['load_checkpoint']
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.test_losses = checkpoint.get('test_losses', [])
            self.det_APs = checkpoint.get('det_AP_history', [])
            self.det_APs_cw = checkpoint.get('det_AP_cw_history', [])
            self.loc_APs = checkpoint.get('loc_AP_history', [])
            self.loc_APs_cw = checkpoint.get('loc_AP_cw_history', [])
            return checkpoint.get('epoch', 0)
        return 0


    def save_network(self, epoch, folder, best, AP):
        """
        Saves the network to the a .pth file.
        Arguments:
            epoch: int - epoch up until the model has been trained to
            folder: string - name of folder in the save_dir
            best: bool - best model until now (e.g. in test loss). It gets saved additionally.
            AP: dict - all the Average Precision statistics + some additional ones
        """
        save_filename = self.name + '.pth'
        save_path = os.path.join(self.save_dir, folder, save_filename)

        if self.use_cuda:
            model_state = self.cpu().state_dict()
            self.to(self.device)
        else:
            model_state = self.cpu().state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'AP': AP,
            'det_AP_history': self.det_APs,
            'det_AP_cw_history': self.det_APs_cw,
            'loc_AP_history': self.loc_APs,
            'loc_AP_cw_history': self.loc_APs_cw,
            'configuration': self.configuration
        }, save_path)

        if best:
            save_path = os.path.join(self.save_dir, folder, self.name + "_best.pth")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_losses': self.train_losses,
                'test_losses': self.test_losses,
                'AP': AP,
                'det_AP_history': self.det_APs,
                'det_AP_cw_history': self.det_APs_cw,
                'loc_AP_history': self.loc_APs,
                'loc_AP_cw_history': self.loc_APs_cw
            }, save_path)