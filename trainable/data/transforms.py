import torch
from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np


class MeshWarp(object):
    """MeshWarp(grid_size=8)

    Interface for piecewise affine transforms.

    Args:
        grid_size (int): The number of vertical and horizontal
            samples of the image. A value of 8 samples an 8x8
            grid of points evenly across the image. Default: 8
    """

    def __init__(self, grid_size=8):
        super().__init__()
        self.__dict__.update(locals())

    def _get_random_element(self, shape):
        raise NotImplementedError

    def __call__(self, image):
        t = type(image)
        if t == torch.Tensor:
            image = image.permute(1, 2, 0).numpy()

        r, c, _ = image.shape

        rows = np.linspace(0, r, self.grid_size)
        cols = np.linspace(0, c, self.grid_size)
        rows, cols = np.meshgrid(rows, cols)
        src = np.dstack([cols.flat, rows.flat])[0]

        # apply random motion
        dst = src + self._get_random_element(src.shape)

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)

        return warp(image, tform, output_shape=image.shape).astype('float32')


class NormalMeshWarp(MeshWarp):
    """NormalMeshWarp(std=1, grid_size=8)

    Args:
        std (int): The standard deviation of the normal
            distribution. A higher value means more
            perturbation.

        grid_size (int): The number of vertical and horizontal
            samples of the image. A value of 8 samples an 8x8
            grid of points evenly across the image. Default: 8
    """
    def __init__(self, std=1, grid_size=8):
        super().__init__(grid_size)
        self.__dict__.update(locals())

    def _get_random_element(self, shape):
        return np.random.normal(0, self.std, shape)


class UniformMeshWarp(MeshWarp):
    """UniformMeshWarp(low=-1, high=1, grid_size=8)

    Args:
        low (int): The minimum random perturbation. i.e., the most
            a grid point could be warped up and to the left
            (in pixels). Default: -1

        high (int): The maximum random perturbation. i.e., the most
            a grid point could be warped down and to the right.
            (in pixels). Default: 1

        grid_size (int): The number of vertical and horizontal
            samples of the image. A value of 8 samples an 8x8
            grid of points evenly across the image. Default: 8
    """
    def __init__(self, low=-1, high=1, grid_size=8):
        super().__init__(grid_size)
        self.__dict__.update(locals())

    def _get_random_element(self, shape):
        return np.random.uniform(self.low, self.high, shape)
