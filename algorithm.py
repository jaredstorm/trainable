
class Algorithm(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DummyAlgorithm(Algorithm):
    """Dummy training algorithm for testing purposes."""

    def __init__(self, validate=False):
        super().__init__()
        self.__dict__.update(locals())

    def __call__(self, model, batch, device):
        y = model(batch.to(device))
        loss = y.mean()
        return {self.key('Mean'): loss.item()}

    def key(self, key):
        return ('Validation ' if self.validate else '') + key
