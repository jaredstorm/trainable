from trainable.test import *


class ValidationManager(object):
    def __init__(self, epoch, frequency=1):
        super().__init__()
        self.__dict__.update(locals())
        self.counter = 0

    ####################################################################################################################
    # Getters/Setters                                                                                                  #
    ####################################################################################################################

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self, epoch):
        return self.epoch

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.reset_counter()

    def get_frequency(self):
        return self.frequency

    def reset_counter(self):
        self.counter = 0

    ####################################################################################################################
    # Call Method                                                                                                      #
    ####################################################################################################################

    def __call__(self, session, data, device):
        self.counter += 1

        if self.counter == self.frequency:
            self.counter = 0

            self.epoch.eval()
            metrics = self.epoch(session, data, device)
            self.epoch.train()

            session.append_metrics(metrics)
            return True
        else:
            return False


def test_validate():
    pass


if __name__ == '__main__':
    test_validate()



