from trainable import test


class TrainingEpoch(object):
    """Training epoch base class.

    Usually an epoch just consists of running data
    through the model in batches. But just in case,
    this base class offers a bit more flexibility.

    Extending this Class:
    To subclass a TrainingEpoch that works fluidly with
    the rest of Trainable, it should:
        1. override the __call__ method
        2. The __call__ method should take at least the
           following parameters:
            a. A session instance
            b. A DataLoader or some other data source
            c. A target device for your data
        3. The __call__ method should at very least:
            a. make a call to the training algorithm, and store
               the results.
            b. call optim.zero_grad() and optim.step() for each
               batch trained on. Algorithms purely run data through
               a model, compute the desired loss, and compute the
               gradient. Surrounding your call to the training algorithm
               with these methods is not only good practice, but mandatory
               for training to work properly.
            c. call visualize.
            d. call update on the loop.

    Example::
    >>> class DefaultTrainingEpoch(TrainingEpoch):
    >>>     def __call__(self, session, data, device):
    >>>         model, optim = session.model, session.optim
    >>>         for batch in data:
    >>>             optim.zero_grad()
    >>>             metrics = self.train(model, batch, device)
    >>>             optim.step()
    >>>
    >>>             session.append_metrics(metrics)
    >>>             self.visualize(session.model, batch, device)
    >>>             self.loop.update(session.epoch, metrics)
    """

    def __init__(self, algorithm=None, loop=None, visualize=None):
        super().__init__()
        self.train = algorithm
        self.loop = loop
        self.visualize = visualize

    def set_algorithm(self, algorithm):
        self.train = algorithm

    def get_algorithm(self):
        return self.train

    def set_visualizer(self, visualizer):
        self.visualize = visualizer

    def get_visualizer(self):
        return self.visualize

    def set_loop(self, loop):
        self.loop = loop

    def get_loop(self):
        return self.loop

    def __call__(self, session, data, device):
        raise NotImplementedError


class DefaultTrainingEpoch(TrainingEpoch):
    """DefaultTrainingEpoch()
    The default behavior for an epoch of training.
    """

    def __call__(self, session, data, device):
        model, optim = session.model, session.optim
        for batch in data:
            optim.zero_grad()
            metrics = self.train(model, batch, device)
            optim.step()

            session.append_metrics(metrics)
            self.visualize(session.model, batch, device)
            self.loop.update(session.epoch, metrics)


def epoch_test():
    test.title("Training Epoch")
