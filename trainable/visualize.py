import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

import os
import shutil

########################################################################################################################
# Things to do                                                                                                         #
########################################################################################################################
# TODO: Flesh out class/method descriptions
# TODO: Make normalization more sensible for Plotter and Saver

class Visualizer(object):
    def __init__(self, frequency=1, batch=True, padding=0):
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
    # Interface Methods                                                          #
    ##############################################################################
    def visualize(self, *args, **kwargs):
        """Wrapper for functional behavior."""
        self(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter == self.frequency:
            self.counter = 0
            if self.batch:
                return self.visualize_batch(*args, **kwargs)
            else:
                return self.visualize_single(*args, **kwargs)
        else:
            return False

    def visualize_batch(self, *args, **kwargs):
        """Implement to visualize outputs from a batch of inputs"""
        raise NotImplementedError

    def visualize_single(self, *args, **kwargs):
        """Implememnt to visualize a single output from a batch of outputs"""
        raise NotImplementedError


class Plotter(Visualizer):
    def __init__(self, frequency=1, batch=True, padding=0):
        super().__init__(frequency, batch, padding)
        self.__dict__.update(locals())

    def visualize_batch(self, model, batch, device):
        with torch.no_grad():
            imgs = model(batch.to(device))

            n_imgs = imgs.size(0)
            row = int(n_imgs ** (1 / 2))

            grid = make_grid(
                imgs,
                row,
                self.padding,
                normalize=True,
                pad_value=255
            )

            plt.axis('off')
            axis = plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
            plt.show()

            model.train()

            return True

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

            return True


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
            )

            self.samples += 1

            model.train()

            return True

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

            return True


class PlotterSaver(Visualizer):
    def __init__(self, image_folder, frequency=1, batch=True, padding=0):
        super().__init__(frequency, batch, padding)
        self.plotter = Plotter(frequency, batch, padding)
        self.saver = Saver(image_folder, frequency, batch, padding)

    def visualize_batch(self, model, batch, device):
        ret1 = self.plotter.visualize_batch(model, batch, device)
        ret2 = self.saver.visualize_batch(model, batch, device)
        return ret1 and ret2

    def visualize_single(self, model, batch, device):
        ret1 = self.plotter.visualize_single(model, batch, device)
        ret2 = self.saver.visualize_single(model, batch, device)
        return ret1 and ret2


