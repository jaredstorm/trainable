from collections import defaultdict

from trainable.test import *

class ValidationManager(object):
    def __init__(self, algorithm, frequency=1):
        super().__init__()
        self.frequency = frequency
        self.validate = algorithm
        self.counter = 0

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.counter = 0

    def set_algorithm(self, algorithm):
        self.validate = algorithm

    def get_algorithm(self):
        return self.validate

    def __call__(self, session, data, device):
        self.counter += 1

        if self.counter == self.frequency:
            self.counter = 0
            metrics = self._validate(session, data, device)
            session.append_metrics(metrics)
            return True
        else:
            return False

    def _validate(self, session, data, device):
        with torch.no_grad():
            model = session.model
            model.eval()
            metrics = defaultdict(lambda: 0.0)

            # Collect metrics
            for batch in data:
                mets = self.validate(model, batch, device)
                for key in mets:
                    metrics[key] += mets[key]

            # Average totals
            for key in metrics:
                metrics[key] = metrics[key] / len(data)

            model.train()
            return metrics


def test_validate():
    pass


if __name__ == '__main__':
    test_validate()



