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
    output['PR_curve'] = torch.zeros((2, steps))
    output['PR_curve_cw'] = torch.zeros((n_classes, 2, steps))

    thresh = torch.linspace(0, 1, steps, device=preds.device)
    preds_thresh = (preds[:, :, None] > thresh).int()
    pr, re, pr_cw, re_cw = precision_recall(preds_thresh, targets, classwise=True)
    output['PR_curve'][0], output['PR_curve'][1] = pr.cpu().numpy(), re.cpu().numpy()
    output['PR_curve_cw'][:, 0], output['PR_curve_cw'][:, 1] = pr_cw.cpu().numpy(), re_cw.cpu().numpy()

    output['PR_curve'][0] = np.maximum.accumulate(output['PR_curve'][0])
    output['PR_curve_cw'][:, 0] = np.maximum.accumulate(output['PR_curve_cw'][:, 0], axis=1)

    return output



class AP_tester:
    def __init__(self, dataset, device):
        self.device = device
        self.dataset_len = len(dataset)
        self.n_classes = dataset.n_classes
        self.all_targets = torch.zeros((self.dataset_len, self.n_classes))
        self.all_bboxes = []
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=device)

        for i in range(len(dataset)):
            target, bboxes = dataset[i][1:]
            self.all_targets[i] = target
            bboxes = [bbox.to(device) for bbox in bboxes]
            self.all_bboxes.append(bboxes)

        self.all_targets = self.all_targets.to(device)

        # index to fill up the outputs
        self.i = 0

    def update(self, new_outputs):
        self.all_outputs[self.i: (self.i + len(new_outputs))] = new_outputs
        self.i += len(new_outputs)

    def run(self):
        AP = detection_AP(torch.sigmoid(self.all_outputs), self.all_targets)
        return AP

    def reset(self):
        self.all_outputs = torch.zeros((self.dataset_len, self.n_classes), device=self.device)
        self.i = 0



