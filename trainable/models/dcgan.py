import torch.nn as nn
import math
import numpy as np

class DCGenerator(nn.Module):
    def __init__(self, img=64, base=16, latent=128):
        super().__init__()
        self.__dict__.update(locals())

        self._build_topology()
        self._init_weights()

    def _build_topology(self):
        nblocks = math.log2(self.img)
        assert nblocks - int(nblocks) == 0, "img must be powers of 2"
        nblocks = int(nblocks)

        in_chan = self.base * 2**(nblocks//2)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(self.latent, in_chan, kernel_size=1))

        # The modulo test ensures that the generator mirrors the critic
        halve = True if nblocks % 2 == 0 else False

        for _ in range(nblocks):
            out_chan = in_chan//2 if halve else in_chan
            self.layers.extend([
                nn.ConvTranspose2d(in_chan, out_chan, kernel_size=4, stride=1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True)
            ])

            in_chan = in_chan//2 if halve else in_chan
            halve = not halve

        self.layers.append(nn.Conv2d(in_chan, 3, kernel_size=9, padding=4))
        self.layers.append(nn.Tanh())

    def _init_weights(self):
        for layer in self.layers:
            if layer.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)

        for layer in self.layers:
            x = layer(x)

        print(x.size())
        return x

    def __len__(self):
        return np.sum([np.prod(p.size()) for p in self.parameters()])



class DCCritic(nn.Module):

    def __init__(self, img=64, base=16):
        super().__init__()
        self.__dict__.update(locals())

        self._build_topology()
        self._init_weights()

    def _build_topology(self):
        nblocks = math.log2(self.img)
        assert nblocks - int(nblocks) == 0, "img must be a power of 2"
        nblocks = int(nblocks)

        self.layers = nn.ModuleList()
        self.layers.extend([
            nn.Conv2d(3, self.base, kernel_size=9, padding=4),
            nn.LeakyReLU(inplace=True),
        ])

        in_chan = self.base
        double = False
        for _ in range(nblocks):
            out_chan = in_chan*2 if double else in_chan
            self.layers.extend([
                nn.Conv2d(in_chan, out_chan, kernel_size=4, stride=2),
                nn.BatchNorm2d(out_chan),
                nn.LeakyReLU(inplace=True)
            ])

            in_chan = in_chan*2 if double else in_chan
            double = not double

        self.layers.append(nn.Conv2d(in_chan, 1, kernel_size=1))
        self.layers.append(nn.Sigmoid())

    def _init_weights(self):
        for layer in self.layers:
            if layer.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            print(x.size())
            x = layer(x)
        return x

    def __len__(self):
        return np.sum([np.prod(p.size()) for p in self.parameters()])
