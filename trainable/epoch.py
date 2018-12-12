from trainable import test
from collections import defaultdict
from statistics import mean


class TrainingEpoch(object):
    """Training epoch base class.

    Usually an epoch just consists of running data
    through the model in batches. But just in case,
    this base class offers a bit more flexibility.

    Subclassing:

    Example::
        >>> class DefaultTrainingEpoch(TrainingEpoch):
        >>>     def __call__(self, session, data, device):
        >>>         model, optim = session.model, session.optim
        >>>         for batch in data:
        >>>             optim.zero_grad()
        >>>             metrics = self.algorithm(model, batch, device)
        >>>             optim.step()
        >>>
        >>>             session.append_metrics(metrics)
        >>>
        >>>             if eval:
        >>>                 self.append_metrics(metrics)
        >>>             else:
        >>>                 self.visualizer(session.model, batch, device)
        >>>                 self.loop.update(session.epoch, metrics)
        >>>
        >>>         if self.eval:
        >>>             averages = self.average_metrics()
        >>>             self.clear_averages()
        >>>             return averages
    """

    def __init__(self, algorithm=None, loop=None, visualizer=None, eval=False):
        super().__init__()
        self.__dict__.update(locals())
        self.metrics = defaultdict(lambda: [])

    def __call__(self, session, data, device):
        """Override This.

        For the override to behave properly:
        1. The __call__ method should take at least the
           following parameters:
            a. A session instance
            b. A DataLoader or some other data source
            c. A target device for your data
        2. Zero your gradients before making the call to the algorithm
        3. Take an optimization step after making the call to the algorithm.
        4. Append metrics from the training algorithm to your session.
        5. If in evaluation mode, save the metrics using self.append_metrics().
        6. If not in evaluation mode, make a call to the visualizer and update the loop.
        7. Once the epoch is finished, if in evaluation mode, calculate the average metrics,
            clear any stored metrics, and return the averages.
        """
        raise NotImplementedError

    ####################################################################################################################
    # Getters/Setters                                                                                                  #
    ####################################################################################################################

    def get_algorithm(self):
        return self.algorithm

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def get_loop(self):
        return self.loop

    def set_loop(self, loop):
        self.loop = loop

    def set_visualizer(self, visualizer):
        self.visualizer = visualizer

    def get_visualizer(self):
        return self.visualizer

    ####################################################################################################################
    # Training/Validation Toggles                                                                                      #
    ####################################################################################################################

    def validate(self):
        """Shift to evalulation mode."""
        self.eval = True

        if type(self.algorithm) in (list, tuple):
            for alg in self.algorithm:
                alg.validate()
        else:
            self.algorithm.validate()

    def train(self):
        """Shift to training mode."""
        self.eval = False
        if type(self.algorithm) in (list, tuple):
            for alg in self.algorithm:
                alg.train()
        else:
            self.algorithm.train()

    ####################################################################################################################
    # Managing Evaluation Metrics                                                                                      #
    ####################################################################################################################

    def append_metrics(self, metrics):
        """Store metrics."""
        for key in metrics:
            self.metrics[key].append(metrics[key])

    def average_metrics(self):
        """Calculate averages"""
        metrics = {}
        for key in self.metrics:
            metrics[key] = mean(self.metrics[key])

        return metrics

    def clear_metrics(self):
        self.metrics = defaultdict(lambda: [])


class DefaultTrainingEpoch(TrainingEpoch):
    """DefaultTrainingEpoch()
    The default behavior for an epoch of training.
    """

    def __call__(self, session, data, device):
        model, optim = session.model, session.optim
        for batch in data:
            optim.zero_grad()
            metrics = self.algorithm(model, batch, device)
            optim.step()

            session.append_metrics(metrics)

            if self.eval:
                self.append_metrics(metrics)
            else:
                self.visualizer(session.model, batch, device)
                self.loop.update(session.epoch, metrics)

        if self.eval:
            averages = self.average_metrics()
            self.clear_metrics()
            return averages


def epoch_test():
    test.title("Training Epoch")
