# Surgical Tool Detection and Localization
## Final Project for the Lecture 'Advanced Machine Learning' at Heidelberg University

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project was to classify and localize surgical instruments in endoscopic images and
videos, to eventually participate in the [SurgToolLoc Challenge](https://surgtoolloc.grand-challenge.org/).
Our approach is based on using a backbone, for example ResNet18, to extract features of the
image and training a convolutional layer with kernel size 1x1 based off of those features. This
results in a heatmap for every tool, that indicates its presence. Due to the network beeing Fully
Convolutional, the spatial information of the tool is preserved and by detecting peaks in the 
heatmaps it can be localized.<br/>
Besides the SurgToolLoc dataset, we mostly relied on the [Cholec80 & M2CAI16](http://camma.u-strasbg.fr/datasets) dataset, provided by the 
<em>CAMMA</em> Research Group at Strasbourg University.<br/>
More information can be found in our project report.



<!-- GETTING STARTED -->
## Getting Started

1. Installing requirements
  ```
  pip install -r requirements.txt
  ```
2. For training or testing a model, a config file is needed (templates can be found in the config folder).
First the datasets need to be defined.<br/>
Necessary are:
* dataset_name: surgtoolloc, cholec80 and m2cai16 (implemented in datasets folder)
* dataset_path: directory of the dataset files
* set: the to use set/split, e.g. **trainval** for training or **test** for testing (folder of same name must exist in the dataset_path)
  ```
  "train_dataset_params": {
        "dataset_name": "cholec80",
        "dataset_path": "/path/to/cholec80_sub",
        "set": "trainval",
  ```
* batch_size: number of samples to process at the same time
* shuffle: shuffle the samples in dataset (recommended for training)
* num_workers: number of workers of the dataloader
* [pin_memory](https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader): recommended for usage with GPU
```
        "loader_params": {
            "batch_size": 20,
            "shuffle": true,
            "num_workers": 2,
            "pin_memory": true
        },
```
* transforms: different augmentations (removing them from the list will remove them at runtime)
  + toTensor: should always be enabled
  + resize: desired size the original image should be resized to
  + flip: probability of the image being horizintally flipped
  + masking: RandomMasking - probability of a tile being masked and the size of tiles
  + rotation: maximum degree of random rotation
  + normalize: normalize every image by the datasets mean and standard deviation (recommended)
  + channelswitcher: shuffle the color channels of an image
```
        "transforms": {
            "toTensor": true,
            "resize": [336, 600],
            "flip": 0.5,
            "masking": [0.5, [24, 24]],
            "rotation": 90,
            "normalize": true,
            "channelswitcher": false
        }
    },
```
After the datasets, the model needs to be defined.<br/>
First start with the structure:
* model_name: currently fullyconv is the only implemented approach (implementation in models/fullyconv_model.py)
* backbone.name: resnet{18, 34, 50, 101}, alexnet, vgg{11, 11_bn, 16_bn}
* strides: stride of the last two conv layers of the backbone (used to control the heatmap size)
* modules: list of modules to include after the backbone (right now only locmap exists, e.g. a lstm layer could be included here later)
* pooling: pooling technique to calculate the class scores from the heatmap (recommended: minmaxpooling)
```
    "model_params": {
        "model_name": "fullyconv",
        "backbone": {
            "name": "resnet18",
            "options": {
                "strides": [1, 1]
            }
        },
        "structure": {
            "modules": ["locmap"],
            "pooling": "minmaxpooling"
        },
```
Training parameters:
* max_epochs: number of epochs
* lr: learning rate of the heatmap-producing conv. layer (1) and the backbone (2)
* lr_milestone: epochs at which to apply lr reduction
* gamma: factor by which to reduce lr
* momentum: momentum of the SGD optimizer
* weight_decay: L2-penalty
```
        "max_epochs": 50,
        "optimizer": {
            "lr": [0.01, 0.01],
            "lr_milestones": [30, 45],
            "gamma": 0.45,
            "momentum": 0.9,
            "weight_decay": 1e-3
        },
```
* checkpoint_path: directory where to save checkpoints during training
* load_checkpoint: path of checkpoint, which should be loaded
```
        "checkpoint_path": "/where/to/save/checkpoints",
        "load_checkpoint": "/continue/with/checkpoint.pth"
    },
```
* printout_freq: number of batches after which the train and test loss is printed
* AP_loc_freq: number of epochs after which to measure the localization AP
```
    "printout_freq": 5,
    "AP_loc_freq": 5
```

<!-- USAGE EXAMPLES -->
## Usage

Training:
```
    python3 train.py path/to/config.json
```

Validation:
```
    python3 validate.py path/to/config.json
```

in colab:
```
    %run train.py path/to/config.json
    %run validate.py path/to/config.json
```


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Our approach is mostly based on the papers [Weakly-supervised
learning for tool localization in laparoscopic videos](https://arxiv.org/abs/1806.05573) and [Weakly supervised convolu-
tional LSTM approach for tool tracking in laparoscopic videos](https://arxiv.org/abs/1812.01366)
* This projects structure is loosely based on the [PyTorchProjectFramework](https://github.com/branislav1991/PyTorchProjectFramework) by branislav1991.