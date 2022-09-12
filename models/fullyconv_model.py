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
        x = self.pooling(x)
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