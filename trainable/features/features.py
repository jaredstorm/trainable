import torch
import torch.nn as nn


class Features(object):
    """Features(feats, device=torch.device('cpu'))

    A set of intermediate DNN activations.

    Can mostly be treated in the same manor as a torch Tensor
    """

    def __init__(self, feats, device=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feats = [f.float().to(self.device) for f in feats]

    def cuda(self):
        self.feats = [f.cuda() for f in self.feats]
        return self

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
            return Features([other - self.feats[i] for i in range(lself)])
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([other[i] - self.feats[i] for i in range(lself)])

    def __sub__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([self.feats[i] - other for i in range(lself)])
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([self.feats[i] - other[i] for i in range(lself)])

    def __radd__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([other + self.feats[i] for i in range(lself)])
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([other[i] + self.feats[i] for i in range(lself)])

    def __add__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([self.feats[i] + other for i in range(lself)])
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([self.feats[i] + other[i] for i in range(lself)])

    def __rmul__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([other * self.feats[i] for i in range(lself)])
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([other[i] * self.feats[i] for i in range(lself)])

    def __mul__(self, other):
        lself = len(self.feats)
        if type(other) == float:
            return Features([self.feats[i] * other for i in range(lself)])
        else:
            lother = len(other)
            self._test_lengths(lself, lother)
            return Features([self.feats[i] * other[i] for i in range(lself)])

    def __truediv__(self, other):
        lself = len(self.feats)
        if type(other) in (float, int):
            return Features([self.feats[i] / other for i in range(lself)])
        else:
            raise ValueError("Division by non float/int")

    def __floordiv__(self, other):
        lself = len(self.feats)
        if type(other) in (float, int):
            return Features([self.feats[i] // other for i in range(lself)])
        else:
            raise ValueError("Division by non float/int")

    def __pow__(self, power):
        assert type(power) in (int, float), f"Featureset -- power must be of type int or float"
        return Features([self.feats[i] ** power for i in range(len(self.feats))])

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
        s = "Features([\n"
        for i, f in enumerate(self.feats):
            s += f"  {i}: {f}" + ("," if i != len(self.feats) - 1 else "") + "\n"
        s += "])"

        return s

    def __getitem__(self, index):
        return self.feats[index]

    def _test_lengths(self, l1, l2):
        assert l1 == l2, f"ModelFeatures length mismatch -- self:{l1}, other:{l2}"


class FeatureExtractor(nn.Module):

    def __init__(self, model, layers=None):
        super().__init__()
        self.__dict__.update(locals())
        self.model = self.model.cuda()
        self.feats = []
        self._register_feats(layers)

    def _register_feats(self, layers):
        for i, m in enumerate(self.model.modules()):
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
        return Features(feats)

    def _copy(self, feats):
        f = []
        for feat in feats:
            f.append(feat.clone())
        return f
