import torch


class Algorithm(object):
    def __init__(self, eval=False):
        super().__init__()
        self.__dict__.update(locals())

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def key(self, key):
        return ('Validation ' if self.eval else '') + key


class DummyAlgorithm(Algorithm):
    """Dummy training algorithm for testing purposes."""

    def __init__(self, eval=False):
        super().__init__()
        self.__dict__.update(locals())

    def __call__(self, model, batch, device):
        batch = batch.to(device)
        y = model(batch)
        loss = torch.mean(y)

        if not self.eval:
            loss.backward()

        return {self.key('Mean'): loss.item()}

