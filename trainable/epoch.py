from trainable import test


class TrainingEpoch(object):
    """Training epoch base class.

    Usually an epoch just consists of running data
    through the model in batches. But just in case,
    this offers a bit more flexibility.
    """

    def __init__(self):
        super().__init__()
        self.__dict__.update(locals())

    def __call__(self, session, data, device):
        raise NotImplementedError


class DefaultTrainingEpoch(TrainingEpoch):
    def __init__(self):
        super().__init__()
        self.loop = None
        self.train = None
        self.visualize = None

    def set_algorithm(self, algorithm):
        self.train = algorithm

    def get_algorithm(self):
        return self.train

    def set_visualizer(self, visualizer):
        self.visualize = visualizer

    def get_visualizer(self):
        return self.visualize

    def __call__(self, session, data, device):
        model, optim = session.model, session.optim
        for batch in data:
            optim.zero_grad()
            metrics = self.train(model, batch, device)
            optim.step()

            session.append_metrics(metrics)
            self.visualize(session.model, batch, device)
            self.loop.update(session.epoch, metrics)


def test_default_epoch():
    test.title("Training Epoch")
