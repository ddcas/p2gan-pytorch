import argparse
import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

from datasets import ContentDataset, content_collate, StyleDataset
from models.vgg import VGG
from models.generator import Generator
from models.discriminator import Discriminator


def main(args):
    logging.getLogger().setLevel(logging.INFO)
    content_path = args.content_path
    style_path = args.style_path
    patch_size = args.patch_size
    patches_per_side = args.patches_per_side
    image_size = patch_size * patches_per_side
    batch_size = args.batch_size
    epochs = args.epochs
    random_seed = args.random_seed
    learning_rate = args.learning_rate
    content_weight = args.content_weight

    logging.info('Checking for CUDA devices...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Manually set random seed for reproducibility')
    torch.manual_seed(random_seed)

    logging.info('Creating content dataloader...')
    # define content images transformations
    content_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])
    dataloader_content = DataLoader(
        ContentDataset(content_path, content_transform),
        batch_size=batch_size, shuffle=True,
        collate_fn=content_collate)

    logging.info('Creating style dataloader...')
    # define style images transformations
    style_transforms = transforms.Compose([
        transforms.ToTensor()])
    dataloader_style = DataLoader(
        StyleDataset(style_path, patch_size, patches_per_side,
                     batch_size*len(dataloader_content), style_transforms),
        batch_size=batch_size, num_workers=0, shuffle=True)

    logging.info('Set up models...')
    vgg = VGG(device).eval()
    generator = Generator().to(device).train()
    discriminator = Discriminator().to(device).train()

    logging.info('Set up optimizers...')
    optim_g = optim.RMSprop(generator.parameters(), lr=learning_rate)
    optim_d = optim.RMSprop(discriminator.parameters(), lr=learning_rate)

    logging.info('Set up losses criteria...')
    content_criterion = nn.MSELoss()
    adv_criterion = nn.BCELoss()

    losses_adv, losses_content = [], []
    # training loop
    for epoch in range(epochs):
        for step, (x, psi) in enumerate(
                zip(dataloader_content, dataloader_style)):
            # forward pass
            phi_x = vgg(x.to(device))
            g_x = generator(x.to(device))
            phi_g_x = vgg(g_x)
            dp_g_x = discriminator(g_x)
            dp_psi = discriminator(psi.to(device))

            # calculate losses
            loss_content = content_criterion(phi_g_x, phi_x)
            loss_adv = adv_criterion(dp_g_x, dp_psi.detach())
            losses_content.append(loss_content.item())
            losses_adv.append(loss_adv.item())
            loss_minmax = loss_adv + content_weight * loss_content

            # update step
            optim_g.zero_grad()
            optim_d.zero_grad()
            loss_minmax.backward()
            optim_g.step()
            optim_d.step()

            # monitor losses
            if step % 128 == 0:
                print('epoch {} - step {} ::: LOSS_ADV: {} --',
                      ' LOSS_CONTENT: {}'.format(epoch, step,
                                                 np.mean(losses_adv),
                                                 np.mean(losses_content)))
                # save generated batch
                save_image(g_x.detach().cpu(),
                           f'./samples/generated_{epoch}-{step}.png')


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
    parser.add_argument('--random-seed', type=int, default=42,
                        help='The seed for the pseudo-random number generators')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='The learning rate for parameter updates')
    parser.add_argument('--content-weight', type=float, default=0.001,
                        help='The weight of the content component of the loss')

    cmd_args = parser.parse_args()

    main(cmd_args)
