import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .transforms import NormalMeshWarp


class UnlabeledImageDataset(Dataset):
    """UnlabeledImageDataset(root, name, download_url=None, transforms=None)

    A Dataset of uncategorized images, such as CelebA.

    Args:
      path (str): The path to the dataset folder.
      transforms (torchvision.transforms.Transform): A torch transformation
        or composition of transformations.
      overfit (int): Shrink the dataset to a specified value, for network
        trainability testing.
    """

    def __init__(self, path, transforms=None, overfit=None):
        super().__init__()

        self.images = ImageFolder(path, transform=transforms)
        self.overfit = overfit

    def __getitem__(self, index):
        if index == len(self.images):
            index -= 1

        image = self.images[index]

        if type(image) == torch.Tensor:
            return image
        else:
            return image[0]

    def __len__(self):
        return len(self.images) if self.overfit is None else self.overfit


class WarpedPairsDataset(UnlabeledImageDataset):
    r"""WarperPairsDataset(path, warpfn, transforms, overfit)

    Pairs original images with a randomly warped version of those images.

    Usage::

      dataset = WarpedPairsDataset('./images', UniformMeshWarp())
      original, warped = dataset[index]

    """

    def __init__(self, path, transforms=None, overfit=None, warpfn=None):
        super().__init__(path, transforms, overfit)
        self.warp = warpfn if warpfn is not None else NormalMeshWarp(intensity=8, grid_size=8)

    def __getitem__(self, index):
        image = super().__getitem__(index)
        return image, torch.tensor(self.warp(image)).permute(2, 0, 1)

