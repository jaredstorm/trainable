import torch
import torch.nn.functional as f
import random


class Algorithm(object):
    """Algorithm()

    Base class for running an algorithm on a batch of data.

    Notes:
    This class is designed to tuck away much of the administrivia
    of training on a batch of data, by automating tasks such as:
    + totalling loss terms and calling backward
    + converting loss terms from tensors to floats
    + switching context from training to validation
    + renaming metrics appropriately when running on validation data

    Suggested steps for subclassing your own algorithm:
    1. Inherit from the base class.
    2. Specify any loss weighting as object attributes defined
       on initialization.
    3. Override the run() method.

    Example::
    >>>	class MeanMSE(Algorithm):
    >>>		def __init__(self, mean_weight=1, mse_weight=1):
    >>> 		super().__init__()
    >>>			self.__dict__.update(locals())
    >>>
    >>> 	def run(self, model, batch, device):
    >>>			x, y = batch
    >>> 		x, y = x.to(device), y.to(device)
    >>>
    >>>			y_tilde = model(x)
    >>>			mse_loss = f.mse_loss(y_tilde, y)*self.mse_weight
    >>>			mean_loss = torch.mean(y_tilde)*self.mean_weight
    >>>
    >>>			metrics = {
    >>>				"Mean" : mean_loss,
    >>>				"MSE" : mse_loss
    >>>			}
    >>>
    >>>			return metrics

    To measure metrics you don't intend to train on:
        Algorithms run in either training or evaluation mode.
    When overriding run(), check if self._eval is set. If it is,
    calculate and return the set of desired extra metrics. If you're
    like me, then you'll be happy to know that you can even do
    this by composing Algorithms, with no extra overhead:

    Example::
    >>> class MyAlgorithms(Algorithm):
    >>>		def __init__(self, a_weight=1, b_weight=1, c_weight=1):
    >>>			super(MyAlgorithms, self).__init__()
    >>>			self.alg_a = AlgorithmA(a_weight)
    >>>			self.alg_b = AlgorithmB(b_weight)
    >>>			self.alg_c = AlgorithmC(c_weight)
    >>>
    >>>		def run(self, model, batch, device):
    >>>			if self._eval:
    >>>				a_mets = self.alg_a(model, batch, device)
    >>>				b_mets = self.alg_b(model, batch, device)
    >>>				c_mets = self.alg_c(model, batch, device)
    >>>
    >>>				# These three dictionaries will be merged and processed automatically.
    >>>				return a_mets, b_mets, c_mets
    >>>			else:
    >>>				metrics = self.alg_a(model, batch, device)
    >>>			    return metrics

    You don't even need to worry about propagating calls to train() and eval(). It just works as intended, as simply as
    possible.
    """

    def __init__(self):
        super(Algorithm, self).__init__()
        self._eval = False
        self._eval_prefix = "Validation "
        self._pre_hooks = []
        self._post_hooks = []
        self._metrics = []

    ####################################################################################################################
    # Public Interface                                                                                                 #
    ####################################################################################################################

    def run(self, *args, **kwargs):
        """Override this method.

        Overriding this method (S.I.C.K.):
        1. Signature - If using default Epoch behavior, your signature should include the following in order:
           a. model: A single torch model, or (less frequently) a list of torch models.
           b. batch: A single torch tensor, list of tensors, or any kind of data needed
            by your algorithm (Typically anything that can be handed down from a DataLoader
            object).
           c. device: The device(s) your data and model need to be on. Often, the model will
            already be on the correct device, but the data will not. Since it's possible
            for the batch to contain non-tensor data, and/or data to be placed
            on more than one device, the task of placing data on the device is left to
            the overrider.
        2. Intention - Place the batch of data on the intended device.
        3. Calculation - Calculate your losses as separate terms, keeping them as tensors.
        4. Keys - Return loss terms in a dictionary, giving appropriate names as keys.
        """
        raise NotImplementedError

    def eval(self):
        """Switch to evaluation mode."""
        self._eval = True

        # Set any composed algorithms to eval as well.
        for key in self.__dict__:
            t = type(self.__dict__[key])
            if issubclass(t, Algorithm) and self.__dict__[key] is not self:
                self.__dict__[key].eval()

        return self

    def train(self):
        """Switch to training mode."""
        self._eval = False

        # Set any composed algorithms to train as well.
        for key in self.__dict__:
            t = type(self.__dict__[key])
            if issubclass(t, Algorithm) and self.__dict__[key] is not self:
                self.__dict__[key].train()

        return self

    def register_pre_hook(self, hook):
        """Add a hook that will be called before running the algorithm."""
        self._pre_hooks.append(hook)

    def register_post_hook(self, hook):
        """Add a hook that will be called after running the algorithm."""
        self._post_hooks.append(hook)

    ####################################################################################################################
    # Automagic Methods                                                                                                #
    ####################################################################################################################

    def __call__(self, *args, **kwargs):
        """Run the algorithm on some data, and prepare any measurements taken."""

        for hook in self._pre_hooks:
            hook()

        losses = self.run(*args, **kwargs)

        for hook in self._post_hooks:
            hook()

        # Simplifies algorithm composition for subclasses
        # by merging lists of dictionaries automatically.
        if type(losses) in (tuple, list):
            losses = self._merge(losses)

        if not self._eval:
            self._backward(losses)

        # Convert to float and rename losses if necessary.
        losses = self._itemize(losses)

        return losses

    def _merge(self, metrics):
        merged = {}
        for mets in metrics:
            merged.update(mets)
        return merged

    def _backward(self, metrics):
        """Run backpropegation on the calculated losses."""
        loss = torch.tensor([0.0])
        for key in metrics:
            loss += metrics[key]
        loss.backward()

    def _itemize(self, metrics):
        """Convert calculated metrics from tensors to floats."""
        itemized = {}

        # Prepend validation prefix if performing validation.
        pre = self._eval_prefix if self._eval else ""

        for key in metrics:
            # Skip over metrics that have already been itemized.
            if type(metrics[key]) == float:
                itemized[key] = metrics[key]
            else:
                itemized[pre + key] = metrics[key].item()
        return itemized


class AlgorithmList(Algorithm):
    """AlgorithmList(*algs)

    A class that mimics some of the basic behavior of a python list, while
    retaining algorithm functionality.
    """

    def __init__(self, *algs):
        super().__init__()
        self._algs = list(algs)

    def run(self, *args, **kwargs):
        return [alg(*args, **kwargs) for alg in self._algs]

    def eval(self):
        super().eval()
        self._algs = [alg.eval() for alg in self._algs]
        return self

    def train(self):
        super().train()
        self._algs = [alg.train() for alg in self._algs]
        return self

    def append(self, alg):
        self._algs.append(alg)

    def extend(self, algs):
        self._algs.extend(algs)

    def insert(self, index, alg):
        self._algs.insert(index, alg)

    def remove(self, alg):
        self._algs.remove(alg)

    def pop(self, index=-1):
        return self._algs.pop(index)

    def clear(self):
        self._algs.clear()

    def index(self, alg, **args):
        return self._algs.index(alg, args['start'], args['end'])

    def count(self, alg):
        return self._algs.count(alg)

    def reverse(self):
        self._algs.reverse()

    def copy(self):
        return self._algs.copy()

    def __getitem__(self, index):
        return self._algs[index]


class MSE(Algorithm):
    """MSE(weight=1)

    An algorithm for computing the mean squared error of predicted data.

    Attributes:
        weight (float): How much to scale the calculated loss.
    """

    def __init___(self, weight=1):
        super(MSE, self).__init__()
        self.weight = weight

    def run(self, model, batch, device):
        """Run MSE over data predictions."""
        x, y = batch[0].to(device), batch[1].to(device)
        return {"MSE": f.mse_loss(model(x), y) * self.weight}


########################################################################################################################
#   TESTING RESOURCES                                                                                                  #
########################################################################################################################
class Mean(Algorithm):
    """Dummy training algorithm for testing purposes."""

    def __init__(self, eval=False):
        super().__init__()
        self.__dict__.update(locals())

    def run(self, model, batch, device):
        batch = batch.to(device)
        return {"Mean": torch.mean(model(batch))}


old = '''
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

    def eval(self):
        self.eval = True

    def train(self):
        self.eval = False


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
'''
