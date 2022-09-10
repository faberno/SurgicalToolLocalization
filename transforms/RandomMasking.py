import torch
from numbers import Number
import numpy as np

class RandomMasking(torch.nn.Module):
    def __init__(self, p_mask, patch_size, value):
        super().__init__()
        if not isinstance(value, (Number, list)):
            raise TypeError("Argument value should be a number or list of numbers.")
        if not isinstance(patch_size, (int, tuple, list)):
            raise TypeError("Argument patch_size should be an int, tuple or list.")
        if not isinstance(p_mask, Number):
            raise TypeError("Argument p_mask should be a number.")
        if p_mask < 0 or p_mask > 1:
            raise TypeError("Masking probability should be between 0 and 1.")
        self.p_mask = p_mask
        if isinstance(patch_size, (tuple, list)):
            self.patch_size = patch_size
        else:
            self.patch_size = (patch_size, patch_size)
        self.value = value

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be masked.

        Returns:
            img (Tensor): Masked Tensor image.
        """
        size = img.shape
        if len(size) == 3:
            img = img.unsqueeze(0)
            size = img.shape
        elif len(size) < 3:
            raise TypeError("Tensor must have 3 or 4 dimensions.")
        reshape = False
        if size[1] == 3:
            reshape = True
            img = torch.permute(img, (0, 2, 3, 1))
            size = img.shape
        B, H, W = size[0:-1]
        if not (H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0):
            raise TypeError("Patch size must fit perfectly in image size.")
        n_vert = H // self.patch_size[0]
        n_hor = W // self.patch_size[1]
        n_patches = (B, n_vert, n_hor)
        masked = torch.from_numpy(np.random.binomial(1, self.p_mask, n_patches).astype(bool))

        blocks = img.view(B, n_vert, self.patch_size[0], n_hor, self.patch_size[1],
                          3).swapaxes(2, 3)
        blocks[masked] = torch.Tensor(self.value)

        img = blocks.swapaxes(2, 3).view(size)

        if reshape:
            img = torch.permute(img, (0, 3, 1, 2))

        return img