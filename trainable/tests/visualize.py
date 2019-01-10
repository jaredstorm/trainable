from unittest import TestCase

import matplotlib.pyplot as plt
import os
import shutil

from trainable.visualize import Plotter, Saver, PlotterSaver
import torch


class PlotterTest(TestCase):
    model = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    device = torch.device('cpu')

    def test_batch(self):
        batch = torch.randn(16, 3, 16, 16)
        plot = Plotter()
        passed = plot(self.model, batch, self.device)
        self.assertTrue(passed)

    def test_padding(self):
        batch = torch.randn(16, 3, 16, 16)
        plot = Plotter(padding=2)
        passed = plot(self.model, batch, self.device)
        self.assertTrue(passed)

    def test_single(self):
        batch = torch.randn(16, 3, 16, 16)
        plot = Plotter(batch=False)
        passed = plot(self.model, batch, self.device)
        self.assertTrue(passed)

    def test_frequency(self):
        plot = Plotter(frequency=3)
        batch = torch.randn(16, 3, 16, 16)

        self.assertFalse(
            plot(self.model, batch, self.device),
            "plotting shouldn't have happened yet"
        )

        self.assertFalse(
            plot(self.model, batch, self.device),
            "plotting shouldn't have happened yet"
        )

        self.assertTrue(
            plot(self.model, batch, self.device),
            "plotting should have happened but didn't"
        )


class SaverTest(TestCase):
    model = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    device = torch.device('cpu')
    batch = torch.randn(16, 3, 16, 16)
    path = './saver_tests'

    @classmethod
    def setUp(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @classmethod
    def tearDown(self):
        for file in os.listdir(self.path):
            self.show(os.path.join(self.path, file))

        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @staticmethod
    def show(path):
        plt.axis('off')
        plt.imshow(plt.imread(path))
        plt.show()

    def test_batch(self):
        save = Saver(self.path)
        save(self.model, self.batch, self.device)
        self.assertTrue(
            save.samples == len(os.listdir(self.path)),
            "sample count is not the same as number of images in folder."
        )

    def test_sample_counting(self):
        save = Saver(self.path)
        for i in range(5):
            save(self.model, self.batch, self.device)
            self.assertTrue(
                save.samples == len(os.listdir(self.path)),
                "sample count is not the same as number of images in folder."
            )

    def test_padded_batches(self):
        save = Saver(self.path, padding=2)
        save(self.model, self.batch, self.device)
        self.assertTrue(
            save.samples == len(os.listdir(self.path)),
            "sample count is not the same as number of images in folder."
        )

    def test_single(self):
        save = Saver(self.path, batch=False)
        save(self.model, self.batch, self.device)
        self.assertTrue(
            save.samples == len(os.listdir(self.path)),
            "sample count is not the same as number of images in folder."
        )

    def test_frequency(self):
        save = Saver(self.path, padding=2)
        save.set_frequency(3)

        self.assertFalse(
            save(self.model, self.batch, self.device),
            "Save occurred too early."
        )

        self.assertFalse(
            save(self.model, self.batch, self.device),
            "Save occurred too early."
        )

        self.assertTrue(
            save(self.model, self.batch, self.device),
            "Save didn't occur."
        )


class PlotterSaverTest(TestCase):
    model = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    device = torch.device('cpu')
    batch = torch.randn(16, 3, 16, 16)
    path = './plotter_saver_tests'

    def test_frequency(self):
        ps = PlotterSaver(self.path, padding=2)
        ps.set_frequency(3)
        self.assertFalse(
            ps(self.model, self.batch, self.device),
            "Save occurred too early."
        )

        self.assertFalse(
            ps(self.model, self.batch, self.device),
            "Save occurred too early."
        )

        self.assertTrue(
            ps(self.model, self.batch, self.device),
            "Save didn't occur."
        )
