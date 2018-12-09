import torch.nn as nn
from torchvision.models import vgg19


class AvgPoolVGG(nn.Module):
    """AvgPoolVGG(vgg=torchvision.models.vgg19)

    A pretrained average-pooling vgg model for feature matching.

    Args:
        vgg: Any of the vgg models in torchvision.models
    """

    def __init__(self, vgg=vgg19):
        super().__init__()
        self.vgg = vgg(pretrained=True).features

        for m in self.vgg.children():
            if isinstance(m, nn.MaxPool2d):
                m = nn.AvgPool2d(kernel_size=2, stride=2)

    def to(self, device):
        self.vgg = self.vgg.to(device)
        return self

    def children(self):
        return self.vgg.children()

    def forward(self, x):
        return self.vgg(x)


class VGG(nn.Module):
    """VGG(vgg=torchvision.models.vgg19)

    A pretrained vgg model for feature matching.

    Args:
        vgg: Any of the vgg models in torchvision.models
    """

    def __init__(self, vgg=vgg19):
        super().__init__()
        self.vgg = vgg(pretrained=True).features

    def to(self, device):
        self.vgg = self.vgg.to(device)
        return self

    def children(self):
        return self.vgg.children()

    def forward(self, x):
        return self.vgg(x)
