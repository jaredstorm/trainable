import torch
import torch.nn as nn


class Features(object):
    """Features(feats, device=torch.device('cpu'))

    A set of intermediate features from a model. The Features
    class should feel and behave similarly to a tensor.

    Args:
        feats (list(torch.Tensor)): A list of Tensors pulled
            from a torch Model's activations.

        device (torch.device): The device the tensors should
            remain on.

    Example::
        >>> feats1 = Features([torch.randn(4, 2**i, 32, 32) for i in range(4)])
        >>> feats2 = Features([torch.randn(4, 2**i, 32, 32) for i in range(4)])
        >>> print(feats1 * feats2)
        >>> Features([
        >>>     (0): (4, 1, 32, 32),
        >>>     (1): (4, 2, 32, 32),
        >>>     (2): (4, 4, 32, 32),
        >>>     (3): (4, 8, 32, 32)
        >>> ])
    """

    def __init__(self, feats, device=torch.device('cpu')):
        super().__init__()
        self.feats = [f.float().to(device) for f in feats]
        self.device = device

    def cuda(self):
        self.feats = [f.cuda() for f in self.feats]
        self.device = torch.device('cuda')
        return self

    def cpu(self):
        self.feats = [f.cpu() for f in self.feats]
        self.device = torch.device('cpu')
        return self

    def to(self, device):
        self.feats = [f.to(device) for f in self.feats]
        self.device = device
        return self

    ##############################################################################
    # Math Operations and Other Silly Dunders                                    #
    ##############################################################################

    def mean(self):
        l = [f.mean() for f in self.feats]
        mean = 0
        for f in l:
            mean += f
        mean /= len(l)
        return mean

    def __rsub__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([other - self.feats[i] for i in range(lself)], self.device)
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([other[i] - self.feats[i] for i in range(lself)], self.device)

    def __sub__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([self.feats[i] - other for i in range(lself)], self.device)
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([self.feats[i] - other[i] for i in range(lself)], self.device)

    def __radd__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([other + self.feats[i] for i in range(lself)], self.device)
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([other[i] + self.feats[i] for i in range(lself)], self.device)

    def __add__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([self.feats[i] + other for i in range(lself)], self.device)
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([self.feats[i] + other[i] for i in range(lself)], self.device)

    def __rmul__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([other * self.feats[i] for i in range(lself)], self.device)
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([other[i] * self.feats[i] for i in range(lself)], self.device)

    def __mul__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([self.feats[i] * other for i in range(lself)], self.device)
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([self.feats[i] * other[i] for i in range(lself)], self.device)

    def __truediv__(self, other):
        lself = len(self.feats)
        if type(other) in (float, int):
            return Features([self.feats[i] / other for i in range(lself)], self.device)
        else:
            raise ValueError("Division by non float/int")

    def __floordiv__(self, other):
        lself = len(self.feats)
        if type(other) in (float, int):
            return Features([self.feats[i] // other for i in range(lself)], self.device)
        else:
            raise ValueError("Division by non float/int")

    def __pow__(self, power):
        assert type(power) in (int, float), f"Featureset -- power must be of type int or float"
        return Features([self.feats[i] ** power for i in range(len(self.feats))], self.device)

    def __eq__(self, other):
        if self is other:
            return True

        if isinstance(other, Features) and len(self) == len(other):
            f = other

            for i, feat in enumerate(self.feats):
                if not torch.equal(feat, f[i]):
                    return False

            return True
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return len(self.feats)

    def __str__(self):
        s = "Features(["
        s += "\n" if len(self.feats) > 0 else ""
        for i, f in enumerate(self.feats):
            s += f"  ({i}): {f.size()}" + ("," if i != len(self.feats) - 1 else "") + "\n"
        s += "])"

        return s

    def __getitem__(self, index):
        return self.feats[index]

    def _test_lengths(self, l1, l2):
        assert l1 == l2, f"ModelFeatures length mismatch -- self:{l1}, other:{l2}"


class FeatureExtractor(nn.Module):
    """FeatureExtractor(model, layers=None, device=torch.device('cpu')

    A wrapper class which extracts intermediate features (activations)
    from a given model.

    Args:
        model (torch.nn.Module): A torch model. At the moment, this
            class is currently untested outside of torchvision's vgg
            models, and the AvgPoolVGG included with trainable.

        layers (list[int]): List of indices for the desired activations.
            Default: All intermediate activations.

        device (torch.device): The device of the model.


    Example::
    >>> extract = FeatureExtractor(AvgPoolVGG(), [7, 16, 24, 35])
    >>> feats1 = extract(torch.randn(8, 3, 224, 224))
    >>> feats2 = extract(torch.randn(8, 3, 224, 224))
    >>> ((feats1 - feats2)**2).mean()
    >>> 1.02589384
    """
    def __init__(self, model, layers=None, device=torch.device('cpu')):
        super().__init__()
        self.__dict__.update(locals())
        self.model = self.model.to(device)
        self.feats = []
        self._register_feats(layers)

    def to(self, t):
        super().to(t)
        self.model = self.model.to(t)
        self.device = t
        return self

    def cuda(self):
        super().cuda()
        self.model = self.model.cuda()
        self.device = torch.device('cuda')
        return self

    def cpu(self):
        super().cpu()
        self.model = self.model.cpu()
        self.device = torch.device('cpu')
        return self

    def _register_feats(self, layers):
        if layers is None:
            layers = range(len(self.model.children()))

        for i, m in enumerate(self.model.children()):
            if isinstance(m, nn.ReLU):
                m.inplace = False

            if i in layers:
                def register():
                    def hook(module, x, y):
                        self.feats.append(y)

                    return hook

                m.register_forward_hook(register())

    def forward(self, *inputs):
        self.feats = []
        self.model(*inputs)
        feats = self._copy(self.feats)
        return Features(feats, device=self.device)

    def _copy(self, feats):
        f = []
        for feat in feats:
            f.append(feat.clone())
        return f