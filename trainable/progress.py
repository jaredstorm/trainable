from tqdm import tqdm
from collections import defaultdict

class ProgressManager(object):
    def __init__(self, display_freq=1):
        super().__init__()
        self.loop = None
        self.frequency = display_freq
        self.counter = 0
        self.metrics = defaultdict(lambda: [])

    ##############################################################################
    # SETTINGS METHODS                                                           #
    ##############################################################################

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.counter = 0
        self._reset_metrics()

    ##############################################################################
    # INTERFACE METHODS                                                          #
    ##############################################################################

    def start(self, total, cur=0):
        total = int(total / self.frequency)
        cur = int(cur / self.frequency)
        self.counter = 0
        self.loop = tqdm(total=total, position=cur, ascii=True)

    def update(self, epoch, metrics=None):
        self.counter += 1

        # Remember metrics between screen updates
        if metrics:
            for key in metrics:
                self.metrics[key].append(metrics[key])

        if self.counter == self.frequency:
            # Report epoch and average together recent metrics history

            description = f"Epoch: {epoch}" + (", " if len(self.metrics) > 0 else "")

            for i, key in enumerate(self.metrics):
                metric = sum(self.metrics[key]) / len(self.metrics[key])
                description += f"{key}: {metric:.2f}"
                description += ", " if i != len(self.metrics) - 1 else ""

            self.loop.update(1)
            self.loop.set_description(description)
            self.counter = 0
            self._reset_metrics()

            return True
        else:
            return False

    def end(self):
        self.loop.close()
        del self.loop
        self.loop = None

    ##############################################################################
    # HELPER METHODS                                                             #
    ##############################################################################

    def _reset_metrics(self):
        self.metrics = defaultdict(lambda: [])