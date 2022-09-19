import torch
from torchmetrics.functional import average_precision
import numpy as np

def precision_recall(out, tar, classwise=False):
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
    def __init__(self, dataset, device, image_size):
        self.device = device
        self.dataset_len = len(dataset)
        self.image_size = image_size
        self.n_classes = dataset.n_classes
        self.all_bboxes = []
        self.all_peaks = []
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=device)
        self.inference = dataset.inference

        if self.inference:
            targets, bboxes = dataset.get_targets()
            # bboxes = [bbox.to(device) for bbox in bboxes]
            self.all_bboxes = bboxes
        else:
            targets = dataset.get_targets()
        self.all_targets = torch.stack(targets).cpu()
        # self.all_targets = self.all_targets.to(device)

        # index to fill up the outputs
        self.i = 0

    def update(self, new_outputs):
        if len(new_outputs)==1:
            self.all_outputs[self.i: (self.i + len(new_outputs))] = new_outputs
            self.i += len(new_outputs)
        else:
            detection_out, localization_out = new_outputs
            self.all_outputs[self.i: (self.i + len(detection_out))] = detection_out
            self.i += len(detection_out)
            for k in range(len(detection_out)):
                mask = localization_out[0][:, 0] == k
                coords = localization_out[0][mask, 1:].cpu()
                vals = localization_out[1][mask].cpu()
                self.all_peaks.append([coords, vals])

    def run(self, compute_AP_loc=False, F1=False):
        if F1:
            AP_det = compute_F1(torch.sigmoid(self.all_outputs), self.all_targets)
        else:
            AP_det = self.detection_AP()
        if self.inference and compute_AP_loc:
            AP_loc = self.localization_AP()
        else:
            AP_loc = dict()
        self.reset()
        return {**AP_det, **AP_loc}

    def reset(self):
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=self.device)
        self.all_peaks = []
        self.i = 0

    def detection_AP(self):
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

    def localization_AP(self, tolerance=16):
        peaks = self.all_peaks
        bboxes = self.all_bboxes
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

            # boxes = torch.vstack((boxes, torch.tensor([[270., 180. ,320., 220.],[50., 50. ,80., 80.]])))
            for t_id, t in enumerate(torch.linspace(0, 1, steps)):
                mask = peaks[i][1] > t
                if len(mask) == 0:
                    for l in box_labels:
                        FN_cw[l, t_id] += 1
                        FN[t_id] += 1
                    continue
                pred_pos = peaks[i][0][:, 1:].fliplr()[mask]
                pred_labels = peaks[i][0][:, 0][mask]

                # box_centers = torch.hstack(((boxes[:, [0]] + boxes[:, [2]]) / 2, (boxes[:, [1]] + boxes[:, [3]]) / 2))
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