import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import math


class Autoencoder(nn.Module):
    """Autoencoder(img=128, base=16, latent=128)

    An autoencoder class whose topology is determined dynamically
    on initialization.

    Attributes:
        img (int): Image size (square). Should be a power of 2.
            Default: 128

        base (int): The number of filters on the first layer
            of the encoder and the last layer of the decoder.
            filter numbers double periodically as they approach
            the middle of the autoencoder. Default: 16.

        latent (int): The size of latent vector encoded by the model.
            Default: 128.
    """

    def __init__(self, img=128, base=16, latent=128):
        super().__init__()

        self.encoder = Encoder(img, base, latent)
        self.decoder = Decoder(img, base, latent)
        self.latent = self.encoder.latent

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def __len__(self):
        """Returns the number of parameters in the model."""
        return np.sum([np.prod(p.size()) for p in self.parameters()])


class Encoder(nn.Module):
    """Encoder(img=128, base=16, latent=128)

    An encoder class whose topology is determined dynamically
    on initialization.

    Attributes:
        img (int): Image size (square). Should be a power of 2.
            Default: 128

        base (int): The number of filters on the first layer
            of the encoder and the last layer of the decoder.
            filter numbers double periodically as they approach
            the middle of the autoencoder. Default: 16.

        latent (int): The size of latent vector encoded by the model.
            Default: 128.
    """

    def __init__(self, img=128, base=16, latent=128):
        super().__init__()

        nblocks = math.log2(img)
        assert nblocks - int(nblocks) == 0, "img must be a power of 2"
        nblocks = int(nblocks)

        self.layers = nn.ModuleList([nn.Conv2d(3, base, kernel_size=9, stride=1, padding=4)])
        double = False
        for _ in range(nblocks):
            out_channels = base * 2 if double else base
            self.layers.extend([
                nn.Conv2d(base, out_channels, kernel_size=3, stride=2, dilation=2, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            ])
            base = base * 2 if double else base
            double = not double

        self.layers.append(nn.Conv2d(base, latent, kernel_size=1))

        for layer in self.layers:
            if layer.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal_(layer.weight)

        self.latent = latent

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def __len__(self):
        return np.sum([np.prod(p.size()) for p in self.parameters()])


class Upsample(nn.Module):
    """Wrapper class for torch.nn.functional.interpolate."""

    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.__dict__.update(locals())

    def forward(self, x):
        return f.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Decoder(nn.Module):
    """Decoder(img=128, base=16, latent=128)

    An decoder class whose topology is determined dynamically
    on initialization.

    Attributes:
        img (int): Image size (square). Should be a power of 2.
            Default: 128

        base (int): The number of filters on the first layer
            of the encoder and the last layer of the decoder.
            filter numbers double periodically as they approach
            the middle of the autoencoder. Default: 16.

        latent (int): The size of latent vector encoded by the model.
            Default: 128.
    """

    def __init__(self, img=128, base=16, latent=128):
        super().__init__()

        nblocks = math.log2(img)
        assert nblocks - int(nblocks) == 0, "img must be powers of 2"
        nblocks = int(nblocks)

        base = base * 2 ** (nblocks // 2)
        self.latent = latent

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(latent, base, kernel_size=1))

        # The modulo test ensures that the decoder mirrors the encoder
        halve = True if nblocks % 2 == 0 else False

        for _ in range(nblocks):
            out_channels = base // 2 if halve else base
            self.layers.extend([
                Upsample(),
                nn.Conv2d(base, out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                # Upsample()
            ])
            base = base // 2 if halve else base
            halve = not halve

        self.image = nn.Conv2d(base, 3, kernel_size=9, stride=1, padding=4)

        for layer in self.layers:
            if layer.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)

        return self.image(z)

    def __len__(self):
        return np.sum([np.prod(p.size()) for p in self.parameters()])
