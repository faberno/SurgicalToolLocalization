import torch
from torchmetrics.functional import average_precision
import numpy as np
from operator import itemgetter
from math import sqrt

def precision_recall(out, tar, classwise=False):
    """
    Calculates precision and recall from the model outputs (detection probabilities) and the
    targets.
    Arguments:
        out: torch.tensor - model outputs
        tar: torch.tensor - class targets
        classwise: bool - Calculate for every class individually
    """
    if not classwise:
        if len(tar.shape) == 2:
            tar = tar[:, :, None]
        TP = torch.minimum(out, tar).sum(dim=(0, 1))
        FN = torch.maximum(tar - out, torch.tensor(0)).sum(dim=(0, 1))
        FP = torch.maximum(out - tar, torch.tensor(0)).sum(dim=(0, 1))

        precision = torch.nan_to_num(TP / (TP + FP), nan=0.0)
        recall = torch.nan_to_num(TP / (TP + FN), nan=0.0)
        return precision, recall
    else:
        minimum = torch.minimum(out, tar[:, :, None])
        TP_cw = minimum.sum(dim=0)
        TP = TP_cw.sum(dim=0)

        maximum = torch.maximum(tar[:, :, None] - out, torch.tensor(0))
        FN_cw = maximum.sum(dim=0)
        FN = FN_cw.sum(dim=0)

        maximum = torch.maximum(out - tar[:, :, None], torch.tensor(0))
        FP_cw = maximum.sum(dim=0)
        FP = FP_cw.sum(dim=0)

        precision = torch.nan_to_num(TP / (TP + FP), nan=0.0)
        precision_cw = torch.nan_to_num(TP_cw / (TP_cw + FP_cw), nan=0.0)

        recall = torch.nan_to_num(TP / (TP + FN), nan=0.0)
        recall_cw = torch.nan_to_num(TP_cw / (TP_cw + FN_cw), nan=0.0)

        return precision, recall, precision_cw, recall_cw

def compute_F1(preds, targets):
    """
    Computes the F1 Score of the model outputs.
    Arguments:
        preds: torch.tensor - model outputs
        targets: torch.tensor - class targets
    """
    preds = preds.cpu()
    output = dict()
    targets = targets.int()

    steps = 100
    output['det_PR_curve'] = np.zeros((2, steps))

    thresh = torch.linspace(0, 1, steps, device=preds.device)
    preds_thresh = (preds[:, :, None] > thresh).int()
    pr, re = precision_recall(preds_thresh, targets, classwise=False)

    F1 = (2 * pr * re) / (pr + re)
    output['det_PR_curve'][0], output['det_PR_curve'][1] = pr.cpu().numpy(), re.cpu().numpy()
    output['F1'] = F1
    output['det_PR_curve'][0] = np.maximum.accumulate(output['det_PR_curve'][0])
    return output


def area_under_curve(x):
    """
    Calculates the area under a precision-recall curve
    """
    if len(x.shape) == 2:  # classwise or not
        if x[1, 0] > x[1, -1]:  # flip if recall is descending
            x = np.fliplr(x)
        spacing = np.diff(x[1])
        area = (spacing * x[0, 1:]).sum()
    else:
        if x[0, 1, 0] > x[0, 1, -1]:
            x = np.flip(x, axis=2)
        spacing = np.diff(x[:, 1], axis=1)
        area = (spacing * x[:, 0, 1:]).sum(axis=1)
    return area



class AP_tester:
    """
    Class to test the models Average Precision and other statistics. While iterating through the
    test batches, it's filled up with the model outputs.
    """
    def __init__(self, dataset, device, image_size, model_strides):
        """
        Arguments:
            dataset: Dataset - Dataset that the model is tested on
            device: string - Device that the model lies on
            image_size: tuple/list - size that the dataset images are resized to
            model_strides: list - Strides of the last two convolutional layers of the network
        """
        self.device = device
        self.dataset_len = len(dataset)
        self.image_size = image_size
        self.n_classes = dataset.n_classes
        self.all_bboxes = []
        self.all_peaks = []
        self.all_peak_values = []
        self.all_indices = []
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=device)
        self.inference = dataset.inference
        self.model_strides = model_strides

        if self.inference:
            targets, bboxes = dataset.get_targets()
            self.all_bboxes = bboxes
        else:
            targets = dataset.get_targets()
        self.all_targets = torch.stack(targets).cpu()
        self.i = 0

    def update(self, new_outputs, indices):
        """
        Add the outputs of the current batch
        Arguments:
            new_outputs: dict - Class scores and found peaks
            indices: Indices of the images in the dataset. Only important if we test on the
                     shuffled dataset
        """
        self.all_indices.append(indices)
        if len(new_outputs['class_scores'].shape) == 2:
            batch_size = len(new_outputs['class_scores'])
        else:
            batch_size = 1
        self.all_outputs[self.i: (self.i + batch_size)] = new_outputs['class_scores']
        if 'peak_list' in new_outputs:
            self.all_peaks.extend(new_outputs['peak_list'])
            self.all_peak_values.extend(new_outputs['peak_values'])
        self.i += batch_size

    def run(self, compute_AP_loc=False, F1=False):
        """
        After iterating through all batches, compute the statistics. Afterwards reset all lists.
        Arguments:
            compute_AP_loc: bool - Compute the localization metrics
            F1: bool - instead of AP, compute the F1-Score.
        """
        self.all_indices = torch.hstack(self.all_indices)
        self.all_bboxes = itemgetter(*self.all_indices)(self.all_bboxes)
        self.all_targets = self.all_targets[self.all_indices]
        if F1:
            AP_det = compute_F1(torch.sigmoid(self.all_outputs), self.all_targets)
        else:
            AP_det = self.detection_AP()
        if self.inference and compute_AP_loc:
            dist_error = self.distance_error()
            AP_loc = self.localization_AP()

        else:
            AP_loc = dict()
            dist_error = dict()
        self.reset()
        return {**AP_det, **AP_loc, **dist_error}

    def reset(self):
        """
        Reset all model outputs
        """
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=self.device)
        self.all_peaks = []
        self.all_peak_values = []
        self.all_indices = []
        self.i = 0

    def detection_AP(self):
        """Compute the PR-Curve and Average Precision of the detection (total and classwise)"""
        preds = torch.sigmoid(self.all_outputs).cpu()
        n_classes = self.all_targets.shape[1]
        output = dict()
        targets = self.all_targets.int()

        output['det_AP'] = average_precision(preds=preds, target=targets,
                                             num_classes=preds.size(1),
                                             average='macro').item()
        output['det_AP_cw'] = torch.zeros(n_classes)
        for c in range(n_classes):
            output['det_AP_cw'][c] = (
                average_precision(preds=preds[:, [c]], target=targets[:, [c]], num_classes=1,
                                  average='macro').item()
            )

        steps = 256
        output['det_PR_curve'] = np.zeros((2, steps))
        output['det_PR_curve_cw'] = np.zeros((n_classes, 2, steps))

        thresh = torch.linspace(0, 1, steps, device=preds.device)
        preds_thresh = (preds[:, :, None] > thresh).int()
        pr, re, pr_cw, re_cw = precision_recall(preds_thresh, targets, classwise=True)
        output['det_PR_curve'][0], output['det_PR_curve'][1] = pr.cpu().numpy(), re.cpu().numpy()
        output['det_PR_curve_cw'][:, 0], output['det_PR_curve_cw'][:,
                                         1] = pr_cw.cpu().numpy(), re_cw.cpu().numpy()

        output['det_PR_curve'][0] = np.maximum.accumulate(output['det_PR_curve'][0])
        output['det_PR_curve_cw'][:, 0] = np.maximum.accumulate(output['det_PR_curve_cw'][:, 0],
                                                                axis=1)

        return output

    def localization_AP(self, tolerance=8):
        """
        Compute the PR-Curve and Average Precision of the localization (total and classwise)
        Arguments:
            tolerance: int - number of pixels that we allow a peak to lie outside a bounding box.
                             Should be the global stride.
        """
        tolerance = tolerance * self.model_strides[0] * self.model_strides[1]
        peaks = self.all_peaks
        peak_values = self.all_peak_values
        bboxes = self.all_bboxes
        assert len(peaks) == len(bboxes)
        steps = 100
        TP_cw, FP_cw, FN_cw = np.zeros((self.n_classes, steps)), np.zeros(
            (self.n_classes, steps)), np.zeros((self.n_classes, steps))
        TP, FP, FN = np.zeros(steps), np.zeros(steps), np.zeros(steps)

        for i in range(len(bboxes)):
            boxes = bboxes[i].clone().int()
            box_labels = boxes[:, 0]
            boxes = boxes[:, 1:]
            boxes[:, :2] = torch.maximum(boxes[:, :2] - tolerance, torch.tensor(0))
            boxes[:, [2]] = torch.minimum(boxes[:, [2]] + tolerance, torch.tensor(self.image_size[1]))
            boxes[:, [3]] = torch.minimum(boxes[:, [3]] + tolerance, torch.tensor(self.image_size[0]))

            for t_id, t in enumerate(torch.linspace(0, 1, steps)):
                mask = peak_values[i] > t
                if len(mask) == 0:
                    for l in box_labels:
                        FN_cw[l, t_id] += 1
                        FN[t_id] += 1
                    continue
                pred_pos = peaks[i][:, 1:].fliplr()[mask]
                pred_labels = peaks[i][:, 0][mask]

                a = boxes[:, None, :2] < pred_pos
                b = boxes[:, None, 2:] > pred_pos
                in_box = torch.all(torch.dstack((a, b)), dim=2)

                same_label = box_labels[:, None] == pred_labels

                match = torch.all(torch.stack((in_box, same_label)), dim=0)
                for bl in np.unique(box_labels.cpu()):
                    mask = box_labels == bl
                    TP_cw[bl, t_id] += torch.count_nonzero(match[mask].any(dim=0)).item()
                    FN_cw[bl, t_id] += torch.count_nonzero(~match[mask].any(dim=1)).item()
                for pl in np.unique(pred_labels.cpu()):
                    mask = pred_labels == pl
                    FP_cw[pl, t_id] += torch.count_nonzero(~match[:, mask].any(dim=0)).item()
                TP[t_id] += torch.count_nonzero(match.any(dim=0)).item()
                FP[t_id] += torch.count_nonzero(~match.any(dim=0)).item()
                FN[t_id] += torch.count_nonzero(~match.any(dim=1)).item()

        precision = np.nan_to_num(TP / (TP + FP), nan=0.0)
        precision = np.maximum.accumulate(precision)
        precision_cw = np.nan_to_num(TP_cw / (TP_cw + FP_cw), nan=0.0)
        precision_cw = np.maximum.accumulate(precision_cw, axis=1)

        recall = np.nan_to_num(TP / (TP + FN), nan=0.0)
        recall_cw = np.nan_to_num(TP_cw / (TP_cw + FN_cw), nan=0.0)

        output = dict()
        output['loc_PR_curve'] = np.stack((precision, recall))
        output['loc_AP'] = area_under_curve(output['loc_PR_curve'])
        output['loc_PR_curve_cw'] = np.hstack((precision_cw[:, None], recall_cw[:, None]))
        output['loc_AP_cw'] = area_under_curve(output['loc_PR_curve_cw'])

        return output

    def distance_error(self):
        """
        Computes the average distance between a bounding box center and all predictions. In percent
        of the diagonal length.
        """
        peak_list = self.all_peaks
        peak_values = self.all_peak_values
        bbox_list = self.all_bboxes
        assert len(peak_list) == len(bbox_list)
        all_distances = []
        all_true_labels = []
        for peaks, peak_values, bboxes in zip(peak_list, peak_values, bbox_list):
            bbox_centers = (bboxes[:, 1:3] + bboxes[:, 3:5]) / 2
            true_peaks = peaks[peak_values > 0.5]
            true_peaks[:, 1:] = true_peaks[:, 1:].fliplr()

            distances = torch.sum((true_peaks[:, None, 1:] - bbox_centers)**2, dim=2).sqrt()
            right_label = true_peaks[:, [0]] == bboxes[:, 0]
            right_label_mask = torch.any(right_label, dim=1)
            min_distances = torch.min(distances[right_label_mask], dim=1).values
            labels = true_peaks[:, 0][right_label_mask]
            all_distances.append(min_distances)
            all_true_labels.append(labels)

        all_distances = torch.hstack(all_distances)
        all_true_labels = torch.hstack(all_true_labels)
        class_distances = torch.zeros(self.n_classes)
        diagonal = sqrt(self.image_size[0]**2 + self.image_size[1]**2)
        for i in range(self.n_classes):
            class_distances[i] = torch.mean(all_distances[all_true_labels == i])
        class_distances = class_distances / diagonal * 100
        out = {
            'distance_cw': class_distances,
            'distance': torch.mean(all_distances) / diagonal * 100
        }
        return out