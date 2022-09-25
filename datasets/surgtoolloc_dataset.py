from datasets.base_dataset import get_transform
from datasets.base_dataset import BaseDataset
import torch
import os
import numpy as np
from PIL import Image
import re


class surgtoollocDataset(BaseDataset):
    """Represents a subdataset of the surgtoolloc challange.

    Input params:
        configuration: Configuration dictionary.
    """

    set_weights = {
        'test': [0.79620563, 0.95943953, 0.74004551, 1.64267677, 0.95661765, 0.45585144, 0.54253545, 2.75635593, 3.22029703, 4.45547945, 1.0408,     1.68087855, 1.10441426, 0.96227811],
        'trainval': [0.78714372, 0.87407407, 0.76986951, 1.21308411, 1.28387735, 0.44164682, 0.45416375, 1.16831683, 1.7192053,  1.64303797, 0.85960265, 1.43267108, 1.23149905, 0.57972309]
    }

    mean = [0.4672, 0.3062, 0.3236]
    std = [0.2305, 0.2114, 0.2191]

    classes = {"needle driver" : 0,
               "monopolar curved scissors": 1,
               "force bipolar": 2,
               "clip applier": 3,
               "tip-up fenestrated grasper": 4,
               "cadiere forceps": 5,
               "bipolar forceps": 6,
               "vessel sealer": 7,
               "suction irrigator": 8,
               "bipolar dissector": 9,
               "prograsp forceps": 10,
               "stapler": 11,
               "permanent cautery hook/spatula": 12,
               "grasping retractor": 13}

    n_classes = len(classes)

    inference = False

    def __init__(self, configuration):
        super().__init__(configuration)

        root = configuration['dataset_path']
        self.image_path = root

        self.weights = self.set_weights[configuration['set']]
        self.set = configuration['set']

        configuration['mean'] = self.mean
        configuration['std'] = self.std
        self.transform = get_transform(configuration)
        self.resize = configuration.get('transforms', {}).get('resize', None)

        self.files = self.read_annotations()

    def __getitem__(self, idx):
        file = self.files[idx]
        target = file['labels']
        img = Image.open(os.path.join(
            self.image_path, self.set, file["file"])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if len(img.shape) == 4:
            img = img.squeeze(0)

        return img, target

    def get_targets(self):
        return [f['labels'] for f in self.files]

    def __len__(self):
        # return the size of the dataset
        return len(self.files)

    def read_annotations(self):
        dataset_elements = []
        files = os.listdir(os.path.join(self.image_path, self.set))

        for file in files:
            class_labels = torch.zeros(self.n_classes)
            match = re.search("\[(.*?)\]", file)
            labels = match.group(1).split("_")
            for l in labels:
                class_labels[int(l)] = 1
            size = [620, 900]
            resize = self.resize
            if resize is not None:
                size = resize
            element = {'file': file,
                       'labels': class_labels,
                       'size': size,}
            dataset_elements.append(element)
        return dataset_elements


