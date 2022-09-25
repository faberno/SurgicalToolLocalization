from datasets.base_dataset import get_transform
from datasets.base_dataset import BaseDataset
import torch
import os
from math import floor
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

class m2cai16Dataset(BaseDataset):
    """Represents the m2cai16 dataset.

    Input params:
        configuration: Configuration dictionary.
    """

    set_weights = {
        'train': [0.39145907, 1.02325581, 1.33333333, 1.11675127, 1.0, 0.96491228, 0.91286307],
        'test': [0.35897436, 0.88421053, 1.3125, 1.0, 1.3125, 1.0, 0.875],
        'trainval': [0.39356984, 1.0, 1.45491803, 1.16776316, 1.05654762, 0.88528678, 0.93421053],
        'val': [0.40882353, 0.99285714, 1.75949367, 1.29906542, 1.19827586, 0.80346821, 1.0]
    }

    mean = [0.3241, 0.2207, 0.2154]
    std = [0.2424, 0.2049, 0.2032]

    classes = {
        "Grasper": 0,
        "Bipolar": 1,
        "Hook": 2,
        "Scissors": 3,
        "Clipper": 4,
        "Irrigator": 5,
        "SpecimenBag": 6
    }

    n_classes = len(classes)

    inference = True

    def __init__(self, configuration):
        super().__init__(configuration)

        root = configuration['dataset_path']
        self.image_path = os.path.join(root, "JPEGImages")
        self.annotation_path = os.path.join(root, "Annotations/")
        self.set_path = os.path.join(root, "ImageSets", "Main")

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
            self.image_path, file["file"] + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if len(img.shape) == 4:
            img = img.squeeze(0)

        out = {
            'idx': torch.tensor(idx),
            'img': img,
            'target': target
        }
        return out

    def __len__(self):
        # return the size of the dataset
        return len(self.files)

    def get_targets(self):
        targets = []
        bboxes = []
        for f in self.files:
            targets.append(f['labels'])
            bboxes.append(torch.stack(f['objects']))
        return targets, bboxes

    def read_annotations(self):
        dataset_elements = []
        filename = os.path.join(self.set_path, self.set + '.txt')
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    name = line.split()[0]
                    class_labels = torch.zeros(self.n_classes)
                    tree = ET.parse(self.annotation_path + name + '.xml')

                    size = tree.find('size')
                    size = (int(size.find('height').text),
                            int(size.find('width').text))


                    all_objects = tree.findall('object')
                    objects = []
                    resize = self.resize
                    for i, obj in enumerate(all_objects):
                        tool_id = self.classes[obj.find('name').text]
                        class_labels[tool_id] = 1

                        bndbox = obj.find('bndbox')
                        ymin = int(bndbox.find('ymin').text)
                        xmin = int(bndbox.find('xmin').text)
                        ymax = int(bndbox.find('ymax').text)
                        xmax = int(bndbox.find('xmax').text)
                        if resize is not None:
                            ymin = floor(ymin * (resize[0] / size[0]))
                            xmin = floor(xmin * (resize[1] / size[1]))
                            ymax = floor(ymax * (resize[0] / size[0]))
                            xmax = floor(xmax * (resize[1] / size[1]))
                        objects.append(torch.tensor([tool_id, xmin, ymin, xmax, ymax], dtype=torch.float))

                    if resize is not None:
                        size = resize
                    element = {'file': name,
                               'labels': class_labels,
                               'size': size,
                               'objects': objects}
                    dataset_elements.append(element)
        else:
            raise ValueError(f'Set file {filename} does not exist')
        return dataset_elements
