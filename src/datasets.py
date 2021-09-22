import torch
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .utils.patch_permutation import patch_permutation


class StyleDataset(Dataset):
    """
    PyTorch Dataset for preprocessing and loading style images
    """

    def __init__(self, img_path, patch_size, patches_per_side, size_dataset,
                 transform):
        self.data = patch_permutation(img_path, patch_size, patches_per_side,
                                      size_dataset)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index]
        sample = self.transform(sample).float()

        return sample

    def __len__(self):
        return len(self.data)


class ContentDataset(Dataset):
    """
    PyTorch VOCDetection Dataset for preprocessing and loading content images
    """

    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = self._download_dataset()

    def _download_dataset(self):
        dataset_train = VOCDetection(
            self.dataset_path, year="2007", image_set="trainval",
            transform=self.transform, download=True)
        dataset_test = VOCDetection(
            self.dataset_path, year="2007", image_set="test",
            transform=self.transform, download=True)
        dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])

        return dataset

    def __getitem__(self, index):
        sample = self.data[index]

        return sample

    def __len__(self):
        return len(self.data)


def content_collate(data):
    """
    Custom collation function to discard dataset labels
    """
    images, annotations = zip(*data)

    return torch.stack(images)
