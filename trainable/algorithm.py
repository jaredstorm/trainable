import torch


class Algorithm(object):
    """Algorithm(eval=False)

    Algorithm Base Class

    Args:
        eval (bool): whether or not this algorithm will
            be used for evaluation/validation

    Extending this class:
    To subclass an Algorithm that works fluidly with the rest
    of Trainable, your implementation should:
    - override the __call__ method
    - within the __call__ method:
        + put the data in your training batch to the
          provided device
        + run the batch through your model
        + compute your loss
        + call loss.backward() if self.eval is False
        + return a dictionary with the following format:
            - { self.key('Loss Name'): loss.item() }
            - NOTE: use self.key() with each of your loss names
              so that the class will automatically handle renaming
              keys in eval mode.

    Example::
        >>> class DummyAlgorithm(Algorithm):
        >>>     def __init__(self, eval=False):
        >>>         super().__init__()
        >>>         self.__dict__.update(locals())
        >>>
        >>>     def __call__(self, model, batch, device):
        >>>         batch = batch.to(device)
        >>>         y = model(batch)
        >>>         loss = torch.mean(y)
        >>>
        >>>         if not self.eval:
        >>>             loss.backward()
        >>>
        >>>         return { self.key('Mean'):loss.item() }
    """

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

