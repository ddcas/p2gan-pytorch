from random import randrange
from PIL import Image
import numpy as np


def extract_patches(img: Image, patch_size: int, num_patches: int) -> list:
    """
    Break an image into randomly extracted patches.

    Parameters
    ----------
    img: Image
        The image that will serve as style source
    patch_size: int
        The size of the patches extracted during the patch permutation
    num_patches: int
        The number of patches per height/width of the resulting style image
    """
    # set extraction limits
    img_size = img.size
    limit_width = img_size[0] - patch_size
    limit_height = img_size[1] - patch_size

    patches = []
    for _ in range(num_patches ** 2):
        # generate random coordinates within image limits
        u = randrange(0, limit_width // 2) * 2
        v = randrange(0, limit_height // 2) * 2
        # extract patch
        patch_area = (u, v, u + patch_size, v + patch_size)
        patch = img.crop(patch_area)
        patches.append(patch)

    return patches


def build_image(patches: list, patches_per_side: int) -> Image:
    """
    Build an image from a list of patches.

    Parameters
    ----------
    patches: list
        The extracted patches that will conform the new style images
    patches_per_side: int
        The number of patches per height/width of the resulting style image
    """
    # create destination image
    patch_size = patches[0].size[0]
    img_size = patch_size * patches_per_side
    img = Image.new('RGB', size=(img_size, img_size))
    # populate destination image with patches
    for i, patch in enumerate(patches):
        img.paste(patch, box=(
            (i % patches_per_side) * patch_size,
            (i // patches_per_side) * patch_size))
        # release patch memory
        del patch

    return img


def patch_permutation(
        src_img_path: str,
        patch_size: int,
        patches_per_side: int,
        num_images: int) -> list:
    """
    Perform patch-permutation as described in https://arxiv.org/abs/2001.07466

    Parameters
    ----------
    src_img_path: str
        The location of the image that will serve as style source
    patch_size: int
        The size of the patches extracted during the patch permutation
    patches_per_side: int
        The number of patches per height/width of the resulting style image
    num_images : int
        The number of images that will conform the style dataset
    """
    src_img = Image.open(src_img_path)
    images = []
    for _ in range(num_images):
        patches = extract_patches(src_img, patch_size, patches_per_side)
        new_img = build_image(patches, patches_per_side)
        images.append(np.array(new_img))

    return images
