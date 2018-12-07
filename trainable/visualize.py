from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

import os
import shutil

from trainable.test import *


class Visualizer(object):
    def __init__(self, frequency=1, batch=True, padding=0, debug=False):
        super().__init__()
        self.__dict__.update(locals())
        self.counter = 0

    ##############################################################################
    # Getters/Setters                                                            #
    ##############################################################################

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.counter = 0

    ##############################################################################
    # Functional API                                                             #
    ##############################################################################

    def __call__(self, model, batch, device):
        self.counter += 1
        if self.counter == self.frequency:
            self.counter = 0
            if self.batch:
                return self.visualize_batch(model, batch, device)
            else:
                return self.visualize_single(model, batch, device)
        else:
            return False

    def visualize_batch(self, model, batch, device):
        raise NotImplementedError

    def visualize_single(self, model, batch, device):
        raise NotImplementedError


class Plotter(Visualizer):
    def __init__(self, frequency=1, batch=True, padding=0):
        super().__init__(frequency, batch, padding)
        self.__dict__.update(locals())

    def visualize_batch(self, model, batch, device):
        with torch.no_grad():
            model = model.to(device).eval()
            imgs = model(batch.to(device))

            n_imgs = imgs.size(0)
            row = int(n_imgs ** (1 / 2))

            grid = make_grid(
                imgs,
                row,
                self.padding,
                normalize=True,
                scale_each=True,
                pad_value=255
            )

            plt.axis('off')
            axis = plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
            plt.show()

            model.train()

            return axis

    def visualize_single(self, model, batch, device):
        with torch.no_grad():
            model = model.to(device).eval()
            imgs = model(batch.to(device))

            img = imgs[0]
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
            plt.axis('off')
            axis = plt.imshow(img.cpu().permute(1, 2, 0).numpy())
            plt.show()
            model.train()

            return axis


class Saver(Visualizer):
    def __init__(self, image_folder, frequency=1, batch=True, padding=0):
        super().__init__(frequency, batch, padding)
        self.__dict__.update(locals())

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        files = os.listdir(image_folder)
        files = [f for f in files if 'sample' in f.lower()]
        self.samples = len(files)

    def visualize_batch(self, model, batch, device):
        with torch.no_grad():
            model.eval()
            model = model.to(device)
            imgs = model(batch.to(device))

            n_imgs = imgs.size(0)
            row = int(n_imgs ** (1 / 2))

            save_image(
                imgs,
                os.path.join(self.image_folder, f'sample-{self.samples}.png'),
                row,
                self.padding,
                normalize=True,
                scale_each=True,
            )

            self.samples += 1

            model.train()

    def visualize_single(self, model, batch, device):
        with torch.no_grad():
            model.eval()

            imgs = model(batch.to(device))
            img = imgs[0].unsqueeze(0)
            save_image(
                img,
                os.path.join(self.image_folder, f'sample-{self.samples}.png'),
                normalize=True
            )
            self.samples += 1

            model.train()


class PlotterSaver(Visualizer):
    def __init__(self, image_folder, frequency=1, batch=True, padding=0):
        super().__init__(frequency, batch, padding)
        self.plotter = Plotter(frequency, batch, padding)
        self.saver = Saver(image_folder, frequency, batch, padding)

    def visualize_batch(self, model, batch, device):
        self.plotter.visualize_batch(model, batch, device)
        self.saver.visualize_batch(model, batch, device)

    def visualize_single(self, model, batch, device):
        self.plotter.visualize_single(model, batch, device)
        self.saver.visualize_single(model, batch, device)


def plotter_test():
    title("plotter")

    model = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = torch.randn(16, 3, 16, 16)

    subtest(1, "Batch Functionality")
    plot = Plotter()
    plot(model, batch, device)

    subtest(2, "Batch With Padding")
    plot = Plotter(padding=2)
    plot(model, batch, device)

    subtest(3, "Single")
    plot = Plotter(batch=False)
    plot(model, batch, device)

    subtest(4, "Different Frequency")
    plot = Plotter(frequency=3)
    passed = True
    passed = passed if not plot(model, batch, device) else False
    passed = passed if not plot(model, batch, device) else False
    evaluate(passed)
    plot(model, batch, device)

    end()


def saver_test():
    if os.path.exists('./saver_tests'):
        shutil.rmtree('./saver_tests')

    def show(path):
        plt.axis('off')
        plt.imshow(plt.imread(path))
        plt.show()

    title("saver")

    model = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu')
    batch = torch.randn(16, 3, 16, 16)

    subtest(1, "Batch Functionality")
    save = Saver("./saver_tests")
    save(model, batch, device)
    passed = True
    passed = passed if save.samples == 1 else False
    passed = passed if len(os.listdir('./saver_tests')) == save.samples else False
    evaluate(passed)
    show('./saver_tests/sample-0.png')

    subtest(2, "Proper Sample Counting")
    passed = True
    save = Saver("./saver_tests")
    passed = passed if save.samples == 1 else False
    save(model, batch, device)
    passed = passed if save.samples == 2 else False
    passed = passed if len(os.listdir('./saver_tests')) == save.samples else False
    evaluate(passed)
    show('./saver_tests/sample-1.png')

    subtest(3, "Padded Batches")
    save = Saver("saver_tests", padding=2)
    save(model, batch, device)
    passed = True if len(os.listdir('./saver_tests')) == save.samples else False
    evaluate(passed)
    show('saver_tests/sample-2.png')

    end()
    shutil.rmtree('./saver_tests')


if __name__ == '__main__':
    saver_test()

