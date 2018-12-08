import torch
import torch.nn as nn
import torch.nn.functional as f

import math

class Autoencoder(nn.Module):
    """Autoencoder(enc, dec)

    Encoder/Decoder composite class.

    Args:
        encoder (torch.nn.Module): An encoder module.
        decoder (torch.nn.Module): A decoder module.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.__dict__.update(locals())

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class Encoder(nn.Module):
    """Encoder(img=32, base=16, latent=128, kernel=3)

    Args:
        img (int): The square image size. Default: 32.
        base (int): The number of filters in the first
            convolutional layer. Each layer thereafter
            doubles the number of filters. Default: 16
        latent (int): The length of the output latent
            encoding. Default: 128
        kernel (int): The square size of each convolution
            kernel. Default: 3
    """

    def __init__(self, img=32, base=16, latent=128, kernel=3):
        super().__init__()
        self.__dict__.update(locals())
        self._init_topology(img, base, latent)

    def _init_topology(self, img, base, latent, kernel):
        """Build the network topology from init args."""

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(3, base, 9, padding=9//2))

        # Append Convolutional Layers/Pooling until spatial
        # dimension reaches 1x1.
        while img > 1:
            self.layers.extend([
                # Convolutional Layer
                nn.Conv2d(base, base, kernel, padding=kernel//2),
                nn.BatchNorm2d(base),
                nn.LeakyReLU(),

                # Convolutional Pooling
                nn.Conv2d(base, base*2, kernel_size=2, stride=2),
                nn.BatchNorm2d(base*2),
                nn.LeakyReLU()
            ])

            # Number of filters doubles with each pooling operation.
            base *= 2
            img //= 2

        # Linear layer transforms features to latent vector.
        self.linear = nn.Linear(base, latent)

    def forward(self, x):
        """Encode an image into latent space."""
        for layer in self.layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)
        return self.linear(x)


class Decoder(nn.Module):
    """Decoder(img=32, base=16, latent=128, kernel=3)

    Args:
        img (int): The square image size. Default: 32.
        base (int): The number of filters in the first
            convolutional layer. Each layer thereafter
            doubles the number of filters. Default: 16
        latent (int): The length of the output latent
            encoding. Default: 128
        kernel (int): The square size of each convolution
            kernel. Default: 3
    """

    def __init__(self, img=32, base=16, latent=128, kernel=3):
        super().__init__()
        self.__dict__.update(locals())
        self._init_topology(img, base, latent, kernel)

    def _init_topology(self, img, base, latent, kernel):
        """Build the network topology from init args."""

        base = base * 2**int(math.log2(img))
        self.linear = nn.Linear(latent, base)

        cur = 1
        self.layers = nn.ModuleList()
        while cur < img:
            base //= 2
            cur *= 2
            self.layers.extend([
                Upsample(),
                nn.Conv2d(base*2, base, kernel, padding=kernel//2),
                nn.BatchNorm2d(base),
                nn.ReLU(),

                nn.Conv2d(base, base, kernel, padding=kernel//2),
                nn.BatchNorm2d(base),
                nn.ReLU()
            ])

        self.layers.append(nn.Conv2d(base, 3, 9, padding=9//2))

    def forward(self, x):
        """Decode an image from latent space."""
        x = self.linear(x).reshape(x.size(0), -1, 1, 1)
        for layer in self.layers:
            x = layer(x)
        return x


class Upsample(nn.Module):
    """Wrapper class for f.interpolate (avoids deprecation warnings)."""
    def forward(self, x):
        return f.interpolate(x, scale_factor=x, mode='nearest')

