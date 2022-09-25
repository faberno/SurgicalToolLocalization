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

class FullyConvModel(BaseModel):
    def __init__(self, config):
        """
        Initialize the model.
        """
        super().__init__(config)
        self.module_list = nn.ModuleList()

        self.name = self.create_name(config)

        backbone_fn = find_module_using_name(config['backbone']['name'])
        self.module_list.append(backbone_fn(pretrained=True, **config['backbone']['options']))

        structure = config['structure']
        for m in structure['modules']:
            module = find_module_using_name(m)
            if m == 'locmap':
                num_features = list(self.module_list[-1].modules())[-2].out_channels
                self.module_list.append(module(num_features, config['n_classes']))

        self.pooling = find_module_using_name(structure['pooling'])

        self.criterion_loss = F.binary_cross_entropy_with_logits
        optim_config = config['optimizer']
        optim_input = [{'params': self.module_list[i].parameters(), 'lr': optim_config['lr'][i]}
                       for i in range(len(self.module_list))]
        self.optimizer = optim.SGD(optim_input,
                                   momentum=optim_config['momentum'],
                                   weight_decay=optim_config['weight_decay'])

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=optim_config['lr_milestones'],
                                                        gamma=optim_config['gamma'])

    def create_name(self, config):
        """
        Create a name to save the model under.
        """
        name = ""
        name += (config['backbone']['name'])
        for m in config['structure']['modules']:
            name += ('-' + m)
        name += ('-' + config['structure']['pooling'])
        strides = config['backbone']['options']['strides']
        name += f"_S{strides[0]}{strides[1]}"
        return name

    def forward(self, input):
        """Run forward pass.
        """
        x = input.clone()
        for m in self.module_list:
            x = m(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        out = self.pooling(x, inference=(not self.training),
                           upsample_size=self.configuration['img_size'])
        return out



    # def forward(self, input, return_crm=False):
    #     """Run forward pass.
    #     """
    #     output = dict()
    #     x = input.clone()
    #     for m in self.module_list:
    #         x = m(x)
    #     if not self.training:
    #         crm = x.clone() # class response maps
    #         crm_original = x.clone()
    #     x = self.pooling(x)
    #     output['class_scores'] = x
    #     if not self.training:
    #         if len(crm.shape) == 3:
    #             crm = crm.unsqueeze(0)
    #         class_found = torch.sigmoid(x) > 0.5
    #         crm[~class_found] = 0
    #         # crm -= torch.amin(crm, dim=(2, 3), keepdim=True)
    #         crm[crm < 0] = 0
    #         crm /= torch.amax(crm, dim=(2, 3), keepdim=True)
    #         crm = torch.nan_to_num(crm)
    #         # crm = F.upsample(crm, size=self.configuration['img_size'], mode='bilinear',
    #         #                  align_corners=True)
    #         peaks = find_peaks(crm, upsample_size=self.configuration['img_size'])
    #         output['peaks'] = peaks
    #         if return_crm:
    #             output['crm'] = crm
    #             output['crm_original'] = crm_original
    #     return output
