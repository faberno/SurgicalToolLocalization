from collections import OrderedDict
from models.base_model import BaseModel
from models import find_module_using_name
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import sys
import torch.optim as optim
from utils import transfer_to_device
from utils.peak_stimulation import peak_stimulation


def _median_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)

def _mean_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)

def _max_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)

def find_peaks(crm, upsample_size, aggregation=False, win_size=3, peak_filter=_median_filter):
    if aggregation:
        peak_list, aggregation = peak_stimulation(crm, return_aggregation=aggregation, win_size=win_size, peak_filter=peak_filter)
    else:
        peak_list = peak_stimulation(crm, return_aggregation=aggregation, win_size=win_size, peak_filter=peak_filter)
    peak_values = crm[peak_list[:, 0], peak_list[:, 1], peak_list[:, 2], peak_list[:, 3]]
    upsample = torch.tensor([upsample_size[0] / crm.shape[2], upsample_size[1] / crm.shape[3]])
    peak_list_upsample = peak_list.float()
    peak_list_upsample[:, 2:] += 0.5
    peak_list_upsample[:, 2:] *= upsample[None, :]
    peak_list = peak_list_upsample.round().int()

    return peak_list, peak_values



class FullyConvModel(BaseModel):
    def __init__(self, config):
        """Initialize the model.
        """
        super().__init__(config)
        self.model = nn.ModuleList()

        self.name = self.create_name(config)

        backbone_fn = find_module_using_name(config['backbone']['name'])
        self.model.append(backbone_fn(pretrained=True, **config['backbone']['options']))

        structure = config['structure']
        for m in structure['modules']:
            module = find_module_using_name(m)
            if m == 'locmap':
                num_features = list(self.model[-1].modules())[-2].out_channels
                self.model.append(module(num_features, config['n_classes']))

        self.pooling = find_module_using_name(structure['pooling'])

        self.criterion_loss = F.multilabel_soft_margin_loss
        optim_config = config['optimizer']
        optim_input = [{'params': self.model[i].parameters(), 'lr': optim_config['lr'][i]}
                       for i in range(len(self.model))]
        self.optimizer = optim.SGD(optim_input,
                                   momentum=optim_config['momentum'],
                                   weight_decay=optim_config['weight_decay'])

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=optim_config['lr_milestones'],
                                                        gamma=optim_config['gamma'])

    def create_name(self, config):
        name = ""
        name += (config['backbone']['name'])
        for m in config['structure']['modules']:
            name += ('-' + m)
        name += ('-' + config['structure']['pooling'])
        if config['structure']['multimaps']:
            name += '_MM'
        strides = config['backbone']['options']['strides']
        name += f"_S{strides[0]}{strides[1]}"
        return name

    def forward(self, input):
        """Run forward pass.
        """
        x = input.clone()
        for m in self.model:
            x = m(x)
        if not self.training:
            crm = x.clone() # class response maps
        x = self.pooling(x)
        if not self.training:
            class_found = torch.sigmoid(x) > 0.5
            crm[~class_found] = 0
            # crm -= torch.amin(crm, dim=(2, 3), keepdim=True)
            crm[crm < 0] = 0
            crm /= torch.amax(crm, dim=(2, 3), keepdim=True)
            # crm = F.upsample(crm, size=self.configuration['img_size'], mode='bilinear',
            #                  align_corners=True)
            peaks = find_peaks(crm, upsample_size=self.configuration['img_size'])
            return x, peaks
        return x

    # def train_minibatch(self, input):
    #     input = transfer_to_device(input[0], self.device)
    #     label = transfer_to_device(input[1], self.device)
    #
    #     output = self.forward(input)
    #     loss = self.criterion_loss(output, label, weight=self.train_weights, reduction='mean')
    #     loss.backward()  # calculate gradients
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()

    def optimize_parameters(self):
        """Calculate gradients and update network weights.
        """
        pass