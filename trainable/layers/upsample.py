import torch.nn as nn
import torch.nn.functional as f


class Upsample(nn.Module):
    """Wrapper class for torch.nn.functional.interpolate."""

    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.__dict__.update(locals())

    def forward(self, x):
        return f.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

