import numpy as np
import sys
from subprocess import Popen, PIPE
import utils
# import visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.nn.functional import interpolate

classes = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]


class Visualizer():
    """This class includes several functions that can display images and print logging information.
    """

    def __init__(self, configuration):
        """Initialize the Visualizer class.

        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.display_id = 0
        self.name = configuration['name']

    def reset(self):
        """Reset the visualization.
        """
        pass

    def plot_current_losses(self, epoch, counter_ratio, loss):
        """Display the current losses on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
            losses: Training losses stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y': [], 'legend': ['Loss']}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([loss.item()])
        x = np.squeeze(
            np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1),
            axis=1)
        y = np.squeeze(np.array(self.loss_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.loss_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.create_visdom_connections()

    def print_current_train_loss(self, epoch, max_epochs, iter, max_iters, loss):
        """Print current losses on console.

        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
        message = f'[epoch: {epoch}/{max_epochs}, iter: {iter}/{max_iters}] Train Loss: {loss:.6f}'
        print(message)  # print the message

    def print_current_epoch_loss(self, epoch=None, max_epochs=None, model=None, plot=True,
                                 AP=None):
        """Print current losses on console.

        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
        message = "\n-------------------------------------------------------------\n"
        if epoch is not None:
            message += f'[epoch: {epoch}/{max_epochs}] Train Loss: {model.train_losses[-1]:.6f} Test Loss: {model.test_losses[-1]:.6f} \n'
        if AP is not None:
            message += f'\nDetection AP: {AP["det_AP"]}, AP(class): {AP["det_AP_cw"].tolist()}'
        if AP is not None:
            if 'loc_AP' in AP:
                message += f'\nLocalization AP: {AP["loc_AP"]}, AP(class): {list(AP["loc_AP_cw"])}'
        message += "\n------------------------------------------------------------\n"
        print(message)  # print the message

        if plot:
            if model is not None:
                plt.plot(model.train_losses, label="train")
                plt.plot(model.test_losses, label="validation")
                plt.title('Loss')
                plt.legend()
                plt.show()

            if AP is not None:
                if 'loc_PR_curve' in AP:
                    classlist = list(model.classes.keys())
                    plt.plot(AP['det_PR_curve'][1], AP['det_PR_curve'][0], label="all", linewidth=3,
                             linestyle='dashed')
                    for i, curve in enumerate(AP['det_PR_curve_cw']):
                        plt.plot(curve[1], curve[0], label=classlist[i])
                    plt.title('Detection Precision-Recall')
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.legend()
                    plt.show()

                if 'loc_PR_curve' in AP:
                    plt.plot(AP['loc_PR_curve'][1], AP['loc_PR_curve'][0], label="all", linewidth=3,
                             linestyle='dashed')
                    for i, curve in enumerate(AP['loc_PR_curve_cw']):
                        plt.plot(curve[1], curve[0], label=classlist[i])
                    plt.title('Localization Precision-Recall')
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.legend()
                    plt.show()

    def plot_validation_images(self, originals, heatmaps, targets=None, peaks=None, bboxes=None,
                               figsize=None, alpha=0.5, save=None):
        """
        Plots all the image overlayed with the class heatmaps, the true targets, bounding boxes
        and found peaks.
        Arguments:
            originals: torch.tensor - Untransformed images
            heatmaps: torch.tensor - Class Response Maps
            targets: torch.tensor - True classes (one hot)
            peaks: list - List of found peaks per instance in batch
            bboxes: list - List of true bounding boxes per instance in batch
            figsize: tuple - Size of the plot figure
            alpha: float (0-1) - Alpha value of the heatmaps in the images
            save: string - Path to save the figure to
        """
        shape = heatmaps.shape[:2]
        max = torch.amax(originals, dim=(2, 3), keepdim=True)
        min = torch.amin(originals, dim=(2, 3), keepdim=True)
        originals = ((originals - min) / (max - min)).permute(0, 2, 3, 1)

        if originals.shape[1] != heatmaps.shape[2] or originals.shape[2] != heatmaps.shape[3]:
            heatmaps = interpolate(heatmaps, originals.shape[1:3], mode='bilinear')

        if figsize is None:
            figsize = ((shape[1] + 1) * 2, shape[0] + 1)
        fig, ax = plt.subplots(shape[0], shape[1] + 1, figsize=figsize)
        for i in range(shape[0]):
            ax[i, 0].imshow(originals[i])
            ax[i, 0].tick_params(left=False,
                                 bottom=False,
                                 labelleft=False,
                                 labelbottom=False)
            if i == 0:
                ax[i, 0].set_title('Image')
            for j in range(1, shape[1] + 1):
                if targets[i, j - 1].item() == 1:
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax[i, j].spines[axis].set_color('red')
                        ax[i, j].spines[axis].set_linewidth(3)
                ax[i, j].imshow(originals[i], cmap='gray')
                ax[i, j].imshow(heatmaps[i, j - 1], alpha=alpha, vmin=0, vmax=1)
                ax[i, j].tick_params(left=False,
                                     bottom=False,
                                     labelleft=False,
                                     labelbottom=False)
                if i == 0:
                    ax[i, j].set_title(classes[j - 1])

        if peaks is not None:
            for i_peak, peak in enumerate(peaks):
                for p in peak:
                    ax[i_peak, p[0] + 1].scatter(p[2], p[1], c='r')

        if bboxes is not None:
            for i_box, box in enumerate(bboxes):
                for b in box:
                    b = b.int()
                    rect = patches.Rectangle((b[1], b[2]), b[3] - b[1], b[4] - b[2],
                                             linewidth=1, edgecolor='b', facecolor='none')
                    ax[i_box, b[0] + 1].add_patch(rect)

        plt.tight_layout()
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
