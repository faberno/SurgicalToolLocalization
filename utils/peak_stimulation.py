# https://github.com/chuchienshu/ultra-thin-PRM/blob/master/prm/functions/peak_stimulation.py

import torch
import torch.nn.functional as F
from torch.autograd import Function

class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter, largest, threshold):
        ctx.num_flags = 4

        # peak finding
        if not largest:
            assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
            offset = (win_size - 1) // 2
            padding = torch.nn.ConstantPad2d(offset, float('-inf'))
            padded_maps = padding(input)
            batch_size, num_channels, h, w = padded_maps.size()
            element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset,
                          offset: -offset]
            element_map = element_map.to(input.device)
            _, indices = F.max_pool2d(
                padded_maps,
                kernel_size=win_size,
                stride=1,
                return_indices=True)
            peak_map = (indices == element_map)
        else:
            batch_size, num_channels, h, w = input.size()
            element_map = torch.arange(0, h * w).long().view(1, 1, h, w)
            element_map = element_map.to(input.device)
            values, indices = F.max_pool2d(
                input,
                kernel_size=(h, w),
                stride=1,
                return_indices=True)
            peak_map = (indices == element_map) & (values != 0)

        # peak filtering
        if peak_filter:
            mask = input >= peak_filter(input)
            mask_mean = (input > torch.mean(input, dim=(2, 3), keepdim=True)) & (input > threshold)
            peak_map = (peak_map & mask & mask_mean)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)

        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                   peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags

def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None, largest=False, threshold=0.8):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter, largest, threshold)