import argparse

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .datasets import ContentDataset, content_collate, StyleDataset


def main(args):
    content_path = args.content_path
    style_path = args.style_path
    patch_size = args.patch_size
    patches_per_side = args.patches_per_side
    image_size = patch_size * patches_per_side
    batch_size = args.batch_size
    epochs = args.epochs

    # define content images transformations
    content_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    # create content dataloader
    dataloader_content = DataLoader(
        ContentDataset(content_path, content_transform),
        batch_size=batch_size, shuffle=True,
        collate_fn=content_collate)

    # define style images transformations
    style_transforms = transforms.Compose([
        transforms.ToTensor()])

    # create style dataloader
    dataloader_style = DataLoader(
        StyleDataset(style_path, patch_size, patches_per_side,
                     batch_size*len(dataloader_content), style_transforms),
        batch_size=batch_size, num_workers=0, shuffle=True)

    # build vgg
    # TODO

    # build generator
    # TODO

    # build discriminator
    # TODO

    # define optimizers
    # TODO

    # training loop
    for epoch in range(epochs):
        for step, (img_c, img_s) in enumerate(
                zip(dataloader_content, dataloader_style)):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-path', type=str, default='.',
                        help='The directory containing the content dataset')
    parser.add_argument('--style-path', type=str, default='',
                        help='The directory containing the style source image')
    parser.add_argument('--patch-size', type=int, default=9,
                        help='The size of the patches conveying the style')
    parser.add_argument('--patches-per-side', type=int, default=24,
                        help='The number of patch permutations per style image')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='The number of data samples per minibatch')
    parser.add_argument('--epochs', type=int, default=3,
                        help='The number of full dataset training passes')

    cmd_args = parser.parse_args()

    main(cmd_args)
