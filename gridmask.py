import torch
import torch.nn as nn
import numpy as np
import math
from scipy.ndimage import rotate

class Grid(object):
    def __init__(self, d_min=20, d_max=80, rotate=90, ratio=0.5, mode=1, prob=1.0):
        self.d_min = d_min
        self.d_max = d_max
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob  # Initial probability

    def set_prob(self, epoch, max_epoch):
        """Linearly increase probability over epochs"""
        self.prob = min(1, epoch / max_epoch)

    def __call__(self, img):
        """Apply GridMask to a given image tensor"""
        if np.random.rand() > self.prob:
            return img  # Skip transformation based on probability

        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, but got {type(img)}")

        _, h, w = img.shape  # Ensure (C, H, W) format

        # Calculate mask dimensions
        hh = int(math.ceil(math.sqrt(h ** 2 + w ** 2)))  # Extend mask size for rotation
        d = np.random.randint(self.d_min, self.d_max)
        mask_size = int(math.ceil(d * self.ratio))

        # Create a grid mask
        mask = np.ones((hh, hh), np.float32)
        st_h, st_w = np.random.randint(d), np.random.randint(d)

        for i in range(-1, hh // d + 1):
            s, t = max(0, d * i + st_h), min(hh, d * i + st_h + mask_size)
            mask[s:t, :] *= 0

        for i in range(-1, hh // d + 1):
            s, t = max(0, d * i + st_w), min(hh, d * i + st_w + mask_size)
            mask[:, s:t] *= 0

        # Rotate mask
        mask = rotate(mask, angle=np.random.randint(self.rotate), reshape=False, mode='nearest')

        # Crop mask back to image size
        h_offset, w_offset = (hh - h) // 2, (hh - w) // 2
        mask = mask[h_offset:h_offset + h, w_offset:w_offset + w]

        # Convert mask to tensor
        mask = torch.from_numpy(mask).float()

        if self.mode == 1:
            mask = 1 - mask  # Invert mask if needed

        mask = mask.expand_as(img)
        return img * mask  # Apply mask to image

class GridMask(nn.Module):
    def __init__(self, d_min=20, d_max=80, rotate=90, ratio=0.4, mode=1, prob=0.8):
        """Torch module for applying GridMask"""
        super(GridMask, self).__init__()
        self.grid = Grid(d_min, d_max, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        """Adjust probability of applying mask over epochs"""
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        """Apply GridMask to a batch or single image"""
        if not self.training:
            return x

        if x.ndimension() == 4:  # Handle batch processing
            return torch.stack([self.grid(img) for img in x])

        return self.grid(x)  # Single image

if __name__ == '__main__':
    import cv2
    from torchvision import transforms

    img = cv2.imread('./data/kvasir/train/image/ckcu8xad600033b5yc78xfyjx.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure correct color order
    img = transforms.ToTensor()(img)  # Convert to tensor

    grid_mask = GridMask()
    masked_img = grid_mask(img)  # Apply GridMask

    # Convert tensor back to OpenCV format
    masked_img = masked_img.mul(255).byte().permute(1, 2, 0)
