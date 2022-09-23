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

    def train_minibatch(self, input):
        data = transfer_to_device(input['img'], self.device)
        label = transfer_to_device(input['target'], self.device)

        output = self(data)
        loss = self.criterion_loss(output, label, weight=self.train_weights, reduction='mean')
        loss.backward()  # calculate gradients
        self.train_batch_losses.append(loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad()

    def test_minibatch(self, input, ap_tester=None):
        data = transfer_to_device(input['img'], self.device)
        label = transfer_to_device(input['target'], self.device)
        if len(data.shape) == 4:
            batch_size = len(data)
        else:
            batch_size = 1
        with torch.no_grad():
            output = self(data, return_crm=True)

            if 'peaks' in output:
                new_peaks = []
                for k in range(batch_size):
                    mask = output['peaks'][0][:, 0] == k
                    coords = output['peaks'][0][mask, 1:].cpu()
                    vals = output['peaks'][1][mask].cpu()
                    new_peaks.append([coords, vals])
                output['peaks'] = new_peaks

            if ap_tester is not None:
                ap_tester.update(output)
            loss = self.criterion_loss(output['class_scores'], label, reduction='sum')
            self.test_batch_losses.append(loss.item())
        return output

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        The implementation here is just a basic setting of input and label. You may implement
        other functionality in your own model.
        """
        self.input = transfer_to_device(input[0], self.device)
        self.label = transfer_to_device(input[1], self.device)
        if not self.training:
            self.bboxes = input[2]

    def forward(self, input):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self):
        """Load and print networks; create schedulers.
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

        # if last_checkpoint > 0:
        #     for s in self.schedulers:
        #         for _ in range(last_checkpoint):
        #             s.step()


    def save_network(self, epoch, folder, best, AP):
        """Save all the networks to the disk.
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


    def load_networks(self, epoch):
        """Load all the networks from the disk.
        """
        for name in self.network_names:
            if isinstance(name, str):
                load_filename = f'{epoch}_net_{name}.pth'
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(f'loading the model from {load_path}')
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)


    def save_optimizers(self, epoch):
        """Save all the optimizers to the disk for restarting training.
        """
        for i, optimizer in enumerate(self.optimizers):
            save_filename = f'{epoch}_optimizer_{i}.pth'
            save_path = os.path.join(self.save_dir, save_filename)

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'test_losses': self.test_losses
            }, save_path)


    def load_optimizers(self, epoch):
        """Load all the optimizers from the disk.
        """
        for i, optimizer in enumerate(self.optimizers):
            load_filename = f'{epoch}_optimizer_{i}.pth'
            load_path = os.path.join(self.save_dir, load_filename)
            print(f'loading the optimizer from {load_path}')
            state_dict = torch.load(load_path)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            optimizer.load_state_dict(state_dict)


    def print_networks(self):
        """Print the total number of parameters in the network and network architecture.
        """
        pass

    def set_requires_grad(self, requires_grad=False):
        """Set requies_grad for all the networks to avoid unnecessary computations.
        """
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret


    def pre_epoch_callback(self, epoch):
        pass


    def post_epoch_callback(self, epoch, visualizer):
        pass


    def export(self):
        """Exports all the networks of the model using JIT tracing. Requires that the
            input is set.
        """
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                export_path = os.path.join(self.configuration['export_path'], f'exported_net_{name}.pth')
                if isinstance(self.input, list): # we have to modify the input for tracing
                    self.input = [tuple(self.input)]
                traced_script_module = torch.jit.trace(net, self.input)
                traced_script_module.save(export_path)


    def get_current_visuals(self):
        """Return visualization images. train.py will display these images."""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret