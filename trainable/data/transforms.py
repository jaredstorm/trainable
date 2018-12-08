import torch
from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np

class MeshWarp(object):

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
    def __init__(self, intensity=4, grid_size=8):
        super().__init__(grid_size)
        self.__dict__.update(locals())

    def _get_random_element(self, shape):
        return np.random.normal(0, self.intensity, shape)


class UniformMeshWarp(MeshWarp):
    def __init__(self, low=-1, high=1, grid_size=8):
        super().__init__(grid_size)
        self.__dict__.update(locals())

    def _get_random_element(self, shape):
        return np.random.uniform(self.low, self.high, shape)
