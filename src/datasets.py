"""
Datasets for training the P$^2$GAN model for efficient neural style transfer.
A single style image is loaded and broken into patches that will transfer the
style into arbitrary real images. As "content/real" training set, the PASCAL
Visual Object Classes Challenge 2007 is used.
"""

import torch
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, ConcatDataset

from utils.patch_permutation import patch_permutation


class StyleDataset(Dataset):
    """
    PyTorch Dataset for preprocessing and loading style images

    Parameters
    ----------
    img_path: str
        The location of the image that serves as style source
    patch_size : int
        The size of the patches extracted during the patch permutation
    patches_per_side : int
        The number of patches per height/width of the resulting style image
    size_dataset : int
        The number of style images to generate from the style source
    transform : torchvision.transforms
        The set of preprocessing operations performed on the style images
    """

    def __init__(self, img_path, patch_size, patches_per_side, size_dataset,
                 transform):
        self.data = patch_permutation(img_path, patch_size, patches_per_side,
                                      size_dataset)
        self.transform = transform

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.data[index]
        sample = self.transform(sample).float()

        return sample

    def __len__(self) -> int:

        return len(self.data)


class ContentDataset(Dataset):
    """
    PyTorch VOCDetection Dataset for preprocessing and loading content images

    Parameters
    ----------
    dataset_path: str
        The location of the VOC dataset
    transform : torchvision.transforms
        The set of preprocessing operations performed on the content images
    """

    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = self._download_dataset()

    def _download_dataset(self) -> Dataset:
        dataset_train = VOCDetection(
            self.dataset_path, year="2007", image_set="trainval",
            transform=self.transform, download=True)
        dataset_test = VOCDetection(
            self.dataset_path, year="2007", image_set="test",
            transform=self.transform, download=True)
        dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])

        return dataset

    def __getitem__(self, index) -> torch.Tensor:

        sample = self.data[index]

        return sample

    def __len__(self) -> int:

        return len(self.data)


def content_collate(data: list) -> torch.Tensor:
    """
    Custom collation function to discard dataset labels
    """
    images, annotations = zip(*data)

    return torch.stack(images)
