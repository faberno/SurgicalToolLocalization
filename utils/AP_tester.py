import torch
from torchmetrics.functional import average_precision
import numpy as np

def precision_recall(out, tar, classwise=False):
    if not classwise:
        TP = torch.minimum(out, tar).sum()
        FN = torch.maximum(tar - out, torch.tensor(0)).sum()
        FP = torch.maximum(out - tar, torch.tensor(0)).sum()
        if (TP + FP) == 0:
            precision = torch.tensor(0.0)
        else:
            precision = (TP / (TP + FP))
        if (TP + FN) == 0:
            recall = torch.tensor(0.0)
        else:
            recall = (TP / (TP + FN))
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

def detection_AP(preds, targets):
    n_classes = targets.shape[1]
    output = dict()
    targets = targets.int()

    output['AP'] = average_precision(preds=preds, target=targets, num_classes=preds.size(1),
                                     average='macro').item()
    output['AP_classwise'] = torch.zeros(n_classes)
    for c in range(n_classes):
        output['AP_classwise'][c] = (
            average_precision(preds=preds[:, [c]], target=targets[:, [c]], num_classes=1,
                              average='macro').item()
        )

    steps = 256
    output['PR_curve'] = np.zeros((2, steps))
    output['PR_curve_cw'] = np.zeros((n_classes, 2, steps))

    thresh = torch.linspace(0, 1, steps, device=preds.device)
    preds_thresh = (preds[:, :, None] > thresh).int()
    pr, re, pr_cw, re_cw = precision_recall(preds_thresh, targets, classwise=True)
    output['PR_curve'][0], output['PR_curve'][1] = pr.cpu().numpy(), re.cpu().numpy()
    output['PR_curve_cw'][:, 0], output['PR_curve_cw'][:, 1] = pr_cw.cpu().numpy(), re_cw.cpu().numpy()

    output['PR_curve'][0] = np.maximum.accumulate(output['PR_curve'][0])
    output['PR_curve_cw'][:, 0] = np.maximum.accumulate(output['PR_curve_cw'][:, 0], axis=1)

    return output

class AP_tester:
    def __init__(self, dataset, device, image_size):
        self.device = device
        self.dataset_len = len(dataset)
        self.image_size = image_size
        self.n_classes = dataset.n_classes
        self.all_targets = torch.zeros((self.dataset_len, self.n_classes))
        self.all_bboxes = []
        self.all_peaks = []
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=device)

        for i in range(len(dataset)):
            target, bboxes = dataset[i][1:]
            self.all_targets[i] = target
            bboxes = [bbox.to(device) for bbox in bboxes]
            self.all_bboxes.append(torch.stack(bboxes))

        self.all_targets = self.all_targets.to(device)

        # index to fill up the outputs
        self.i = 0

    def update(self, new_outputs):
        if len(new_outputs)==1:
            self.all_outputs[self.i: (self.i + len(new_outputs))] = new_outputs
            self.i += len(new_outputs)
        else:
            detection_out, localization_out = new_outputs
            self.all_outputs[self.i: (self.i + len(detection_out))] = detection_out
            self.i += len(new_outputs)
            for k in range(len(detection_out)):
                mask = localization_out[0][:, 0] == k
                coords = localization_out[0][mask, 1:]
                vals = localization_out[1][mask]
                self.all_peaks.append([coords, vals])


    def run(self):
        AP_det = detection_AP(torch.sigmoid(self.all_outputs), self.all_targets)
        AP_loc = self.localization_AP(self.all_peaks, self.all_bboxes)
        return AP_det

    def reset(self):
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=self.device)
        self.i = 0

    def localization_AP(self, peaks, bboxes, tolerance=8):
        steps = 256
        TP_cw, FP_cw, FN_cw = np.zeros((self.n_classes, steps)), np.zeros(
            (self.n_classes, steps)), np.zeros((self.n_classes, steps))
        TP, FP, FN = np.zeros(steps), np.zeros(256), np.zeros(steps)

        for i in range(len(bboxes)):
            print(i)
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
                        FN_cw[l, t] += 1
                        FN[t] += 1
                    continue
                pred_pos = peaks[i][0][:, 1:].fliplr()[mask]
                pred_labels = peaks[i][0][:, 0][mask]

                # box_centers = torch.hstack(((boxes[:, [0]] + boxes[:, [2]]) / 2, (boxes[:, [1]] + boxes[:, [3]]) / 2))
                a = boxes[:, None, :2] < pred_pos
                b = boxes[:, None, 2:] > pred_pos
                in_box = torch.all(torch.dstack((a, b)), dim=2)

                same_label = box_labels[:, None] == pred_labels

                match = torch.all(torch.stack((in_box, same_label)), dim=0)
                for bl in np.unique(box_labels):
                    mask = box_labels == bl
                    TP_cw[bl, t_id] += torch.count_nonzero(match[mask].any(dim=0)).item()
                    FN_cw[bl, t_id] += torch.count_nonzero(~match[mask].any(dim=1)).item()
                for pl in np.unique(pred_labels):
                    mask = pred_labels == pl
                    FP_cw[pl, t_id] += torch.count_nonzero(~match[:, mask].any(dim=0)).item()
                TP[t_id] += torch.count_nonzero(match.any(dim=0)).item()
                FP[t_id] += torch.count_nonzero(~match.any(dim=0)).item()
                FN[t_id] += torch.count_nonzero(~match.any(dim=1)).item()

        precision = np.nan_to_num(TP / (TP + FP), nan=0.0)
        precision_cw = np.nan_to_num(TP_cw / (TP_cw + FP_cw), nan=0.0)

        recall = np.nan_to_num(TP / (TP + FN), nan=0.0)
        recall_cw = np.nan_to_num(TP_cw / (TP_cw + FN_cw), nan=0.0)

        output = dict()
        output['PR_curve'] = np.stack((precision, recall))
        output['PR_curve_cw'] = a = np.hstack((precision_cw[:, None], recall_cw[:, None]))

        output['PR_curve'][0] = np.maximum.accumulate(output['PR_curve'][0])
        output['PR_curve_cw'][:, 0] = np.maximum.accumulate(output['PR_curve_cw'][:, 0], axis=1)

        return output

def box_point_AP(preds, targets, image_shape, classwise=True, n_classes=7, tolerance=8):
    if classwise and (n_classes is not None):
        TP, FP, FN = np.zeros((n_classes, 256)), np.zeros((n_classes, 256)), np.zeros((n_classes, 256))
    else:
        TP, FP, FN = np.zeros(256), np.zeros(256), np.zeros(256)
    for i in range(len(preds)):
        boxes = targets[i]['boxes'].clone().numpy()
        boxes[:, :2] = np.maximum(boxes[:, :2] - tolerance, 0)
        boxes[:, [2]] = np.minimum(boxes[:, [2]] + tolerance, image_shape[1])
        boxes[:, [3]] = np.minimum(boxes[:, [3]] + tolerance, image_shape[1])
        box_labels = targets[i]['labels'].numpy()
        for t in range(256):
            mask = preds[i]['val'].cpu().numpy() > t
            if len(mask) == 0:
                for l in box_labels:
                    FN[l, t] += 1
                continue
            pred_pos = preds[i]['pos'][:, -2:].flip(dims=(1,)).cpu().numpy()[mask]
            pred_labels = preds[i]['pos'][:, 1].cpu().numpy()[mask]
            # pred_vals = preds[i]['val'].numpy()
            # box_centers = torch.hstack(((boxes[:, [0]] + boxes[:, [2]]) / 2, (boxes[:, [1]] + boxes[:, [3]]) / 2))
            a = boxes[:, None, :2] < pred_pos
            b = boxes[:, None, 2:] > pred_pos
            in_box = np.all(np.dstack((a, b)), axis=2)

            same_label = box_labels[:, None] == pred_labels

            match = np.all((in_box, same_label), axis=0)
            if classwise:
                for c in np.unique(box_labels):
                    mask = box_labels == c
                TP[c, t] += np.count_nonzero(match[mask].any(axis=0))
                FP[c, t] += np.count_nonzero(~match[mask].any(axis=0))
                FN[c, t] += np.count_nonzero(~match[mask].any(axis=1))
            else:
                TP[t] += np.count_nonzero(match.any(axis=0))
                FP[t] += np.count_nonzero(~match.any(axis=0))
                FN[t] += np.count_nonzero(~match.any(axis=1))


