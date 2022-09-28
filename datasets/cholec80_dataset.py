from datasets.base_dataset import get_transform
from datasets.base_dataset import BaseDataset
import torch
import os
import numpy as np
from PIL import Image
import re
from math import floor
import xml.etree.ElementTree as ET


class cholec80Dataset(BaseDataset):
    """Represents a subdataset of the cholec80 dataset.

    Input params:
        configuration: Configuration dictionary.
    """

    set_weights = {
        'trainval': [0.23822441, 1.0, 0.68124118, 1.05805038, 1.05689278, 1.02006336, 0.77218225],
        'test': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    train_mean = [0.3527, 0.2261, 0.2212]
    train_std = [0.2591, 0.2160, 0.2128]

    test_mean = [0.3879, 0.2677, 0.2612]
    test_std = [0.2191, 0.1958, 0.1952]

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
        self.image_path = root
        self.annotation_path = os.path.join(root, "Annotations/")
        self.set_path = os.path.join(root, "ImageSets", "Main")

        self.weights = self.set_weights[configuration['set']]
        self.set = configuration['set']

        if self.set == 'trainval':
            self.mean = self.train_mean
            self.std = self.train_std
            configuration['mean'] = self.train_mean
            configuration['std'] = self.train_std
        else:
            self.mean = self.test_mean
            self.std = self.test_std
            configuration['mean'] = self.test_mean
            configuration['std'] = self.test_std

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

        out = {
            'idx': torch.tensor(idx),
            'img': img,
            'target': target
        }

        return out

    def get_targets(self):
        if self.set == 'test':
            targets = []
            bboxes = []
            for f in self.files:
                targets.append(f['labels'])
                bboxes.append(torch.stack(f['objects']))
            return targets, bboxes
        else:
            return [f['labels'] for f in self.files]

    def __len__(self):
        # return the size of the dataset
        return len(self.files)

    def read_annotations(self):
        dataset_elements = []

        if self.set == 'trainval':
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
        else:
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
                            objects.append(
                                torch.tensor([tool_id, xmin, ymin, xmax, ymax], dtype=torch.float))

                        if resize is not None:
                            size = resize
                        element = {'file': name + ".jpg",
                                   'labels': class_labels,
                                   'size': size,
                                   'objects': objects}
                        dataset_elements.append(element)
            else:
                raise ValueError(f'Set file {filename} does not exist')
            return dataset_elements

