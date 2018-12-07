from trainable.progress import ProgressManager
from trainable.validate import ValidationManager
from trainable.visualize import Plotter, Saver
from trainable.epoch import DefaultTrainingEpoch
from trainable.algorithm import DummyAlgorithm
from trainable.session import AutoSession

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import os
from trainable import test
import shutil


class Trainer(object):
    """Trainer(**args)

    Args:

    1. Default Behaviors
      Sessions:
      - Each session contains:
        + The name of the session
        + A detailed description of the session
        + The metrics stored for the session
        + The most recently completed epoch of training (resuming a session starts
          at session.epoch+1)
        + The model and optimizer states

      Progress Updates:
      - The progress manager displays the epoch and losses after every batch

      Visualization:
      - Displays an image after every batch

      Epoch:
      - Runs all the provided data through the model in batches,
      - Optimizes after each batch
      - Appends the metrics yielded from each batch of training

      Validation:
      - Performs validation after every epoch of training
      - Runs all provided validation data through the model in batches
      - Returns the average metrics for the validation step


    2. Configurable Elements:
      Visualization:
      - self.visualize may be any inheritor of the Visualizer class

      Epoch:
      - Though the default epoch runner will work for most cases, self.epoch may
        be any inheritor of the TrainingEpoch class

      Validation:
      - You may use any inheritor of the Algorithm class to do validation

      Training:
      - You may use any inheritor of the Algorithm class to train on indvidual
        batches


    3. Tunable Behaviors:
      All of the following may be set to your liking:
      - The frequency of progress updates, in batches
      - The frequency of visualization, in batches
      - The frequency of session saving, in epochs
      - The frequency of validation, in epochs


    4. Workflow:
      A Typical workflow looks like the following:

        model = YourModel()
        model = model.to(device)
        optim = YourOptim(model.parameters(), ...)
        train_data = DataLoader(training_dataset, ...)
        test_data = DataLoader(testing_dataset, ...)

        trainer = Trainer(
          visualizer=YourVisualizer(),  # Typically Plotter() or Saver()
          train_alg=YourTrainingAlgorithm(),
          test_alg=YourTrainingAlgorithm(validate=True)
          display_freq=...,
          visualize_freq=...,
          autosave_freq=...,
          validate_freq=...,
          device=...
        )

        trainer.new_session(model, optim, path)
        trainer.name_session('Name')
        trainer.describe_session('Description')
        trainer.train(train_data, test_data, epochs)
    """

    def __init__(self, **args):
        super().__init__()
        self.loop = ProgressManager()
        self.epoch = args.pop('training_epoch', DefaultTrainingEpoch())
        self.epoch.set_visualizer(args.pop('visualizer', Plotter()))

        self.epoch.loop = self.loop
        self.epoch.train = args.pop('train_alg', DummyAlgorithm())
        self.validation = ValidationManager(args.pop('test_alg', DummyAlgorithm(eval=True)))

        self.set_display_frequency(args.pop('display_freq', 1))
        self.set_visualize_frequency(args.pop('visualize_freq', 1))
        self.set_validation_frequency(args.pop('validate_freq', 1))

        self.autosave_freq = args.pop('autosave_freq', 1)
        self.device = args.pop('device', torch.device('cpu'))
        self.session = None

    ##############################################################################
    # Training API                                                               #
    ##############################################################################

    def train(self, train_data, test_data, epochs):
        """Train a model on a set of data.

        Args:
          train_data (torch.utils.data.DataLoader): The DataLoader containing the
            training data
          test_data (torch.utils.data.DataLoader): The DataLoader containing the
            validation data
          epochs (int): How many epochs to train for

        Returns:
            (dict): A dictionary containing titled metrics measured throughout training
        """
        if self.session is None:
            raise AssertionError('No session found. Use "start_session, load_session, or new_session before training')

        self.loop.start(total=len(train_data)*epochs)
        self.session.resume()

        for _ in range(epochs):
            self.session.next_epoch()
            self.validation(self.session, test_data, self.device)
            self.epoch(self.session, train_data, self.device)
            self.session.save()

        self.loop.end()
        self.session.suspend()

        return self.session.metrics

    ##############################################################################
    # Session Management                                                         #
    ##############################################################################

    def start_session(self, model, optim, path):
        """Open up either an existing session or a new session.

        Args:
          model (torch.nn.Module): The model that you'll be training
          optim (torch.optim.Optimizer): The optimizer for the model's parameters.
          path (str): Either the path to save the session to, or the path to
            load the session from.
        """

        if not os.path.exists(path):
            self.new_session(model, optim, path)
        else:
            self.load_session(model, optim, path)

    def new_session(self, model, optim, path):
        """Begin a new training session."""
        self.session = AutoSession(model, optim, path)
        self.session.set_frequency(self.autosave_freq)

    def load_session(self, model, optim, path):
        """Load an existing training session."""
        self.session = AutoSession(model, optim)
        self.session.load(path)
        self.session.set_frequency(self.autosave_freq)

    def name_session(self, name):
        """Give your session a name."""
        self.session.rename(name)

    def describe_session(self, description):
        """Give your session a more in depth description."""
        self.session.describe(description)

    def get_session(self):
        return self.session

    ##############################################################################
    # Configurable Elements                                                      #
    ##############################################################################

    def set_train_algorithm(self, algorithm):
        """Inject a different algorithm for batch training."""
        self.epoch.train = algorithm

    def set_epoch(self, epoch_manager):
        """Inject a different epoch runner."""
        algorithm = self.epoch.get_algorithm()

        self.epoch = epoch_manager
        self.epoch.set_algorithm(algorithm)

        self.epoch.loop = self.loop

    def set_val_algorithm(self, algorithm):
        """Inject a different algorithm for validation of each batch."""
        self.validation.validate = algorithm

    def set_validation(self, validation_manager):
        """Inject a different validation runner."""
        algorithm = self.validation.get_algorithm()
        self.validation = validation_manager
        self.validation.set_algorithm(algorithm)

    def set_visualizer(self, visualizer):
        """Inject a different visualization strategy."""
        self.visualize = visualizer

    ##############################################################################
    # Tunable Behaviors                                                          #
    ##############################################################################

    def to(self, device):
        self.device = device

    def set_display_frequency(self, frequency):
        """Set how often the progress manager updates loss information in batches."""
        self.loop.set_frequency(frequency)

    def set_visualize_frequency(self, frequency):
        """Set how often to sample visuals, in batches."""
        self.epoch.visualize.frequency = frequency

    def set_autosave_frequency(self, frequency):
        """Set how often the session saves in epochs."""
        self.autosave_freq = frequency
        self.session.set_frequency(frequency)

    def set_validation_frequency(self, frequency):
        """Set how often validation is performed, in epochs."""
        self.validation.set_frequency(frequency)


def trainer_test():
    if os.path.exists('./trainer_test'):
        shutil.rmtree('./trainer_test')
    test.title("Trainer")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = test.Model().to(device)
    optim = Adam(model.parameters(), 1e-4)
    train_data = DataLoader(test.Dataset(), batch_size=4)
    test_data = DataLoader(test.Dataset(), batch_size=4)

    # set up training session
    trainer = Trainer(
        train_alg=DummyAlgorithm(),
        test_alg=DummyAlgorithm(eval=True),
        visualizer=Saver('./trainer_test/samples'),
        display_freq=100,
        visualize_freq=100,
        autosave_freq=1,
        validate_freq=1
    )

    trainer.to(device)
    trainer.start_session(model, optim, './trainer_test/test.sesh')
    trainer.name_session('TEST')
    trainer.describe_session('THIS IS A TEST')
    trainer.train(train_data, test_data, 10)

    # TEST 1
    test.subtest(1, "Basic Session Functionality")
    session = trainer.get_session()
    passed = True
    passed = passed if session.epoch == 10 else False
    passed = passed if session.name == 'TEST' else False
    passed = passed if session.description == 'THIS IS A TEST' else False
    passed = passed if 'Mean' in session.metrics else False
    passed = passed if 'Validation Mean' in session.metrics else False
    passed = passed if session.optim == optim else False
    passed = passed if session.model == model else False
    passed = passed if len(os.listdir('trainer_test/samples')) == 25 else False
    test.evaluate(passed)

    test.subtest(2, "Session Saved")
    model = test.Model().to(device)
    optim = Adam(model.parameters(), 1e-4)
    trainer = Trainer(
        train_alg=DummyAlgorithm(),
        test_alg=DummyAlgorithm(eval=True),
        visualizer=Saver('./trainer_test/samples'),
        display_freq=100,
        visualize_freq=100,
        autosave_freq=1,
        validate_freq=1
    )
    trainer.to(device)

    # TEST 2
    trainer.start_session(model, optim, './trainer_test/test.sesh')
    session = trainer.get_session()
    passed = True
    passed = passed if session.epoch == 10 else False
    passed = passed if session.name == 'TEST' else False
    passed = passed if session.description == 'THIS IS A TEST' else False
    passed = passed if 'Mean' in session.metrics else False
    passed = passed if 'Validation Mean' in session.metrics else False
    passed = passed if session.optim == optim else False
    passed = passed if session.model == model else False
    test.evaluate(passed)

    # TEST 3
    test.subtest(3, "Continued Training")
    trainer.train(train_data, test_data, 10)
    test.evaluate(session.epoch == 20)
    test.end()

    if os.path.exists('./trainer_test'):
        shutil.rmtree('./trainer_test')


if __name__ == '__main__':
    trainer_test()
