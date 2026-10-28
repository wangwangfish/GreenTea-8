"""
Data Augmentation and Transforms
=================================
Standard data preprocessing pipeline for the GreenTea-8 dataset.

Transforms:
----------
Training:
    RandomResizedCrop(224) → RandomHorizontalFlip → ToTensor → Normalize
Validation:
    Resize(256) → CenterCrop(224) → ToTensor → Normalize

ImageNet Normalization (Standard):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

Reference:
----------
He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
"""

from typing import Tuple

import torchvision.transforms as T


# ImageNet standard normalization values
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(
    input_size: int = 224,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.333),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    horizontal_flip_prob: float = 0.5,
) -> T.Compose:
    """Get training data transforms.

    Standard ImageNet-style training augmentation:
    1. RandomResizedCrop: Randomly crop and resize to target size
    2. RandomHorizontalFlip: Mirror with 50% probability
    3. ToTensor: Convert PIL Image to tensor [0, 1]
    4. Normalize: Standardize with ImageNet mean/std

    Args:
        input_size: Target image size. Default: 224.
        scale: Range of area proportions to crop. Default: (0.08, 1.0).
        ratio: Range of aspect ratios to use. Default: (0.75, 1.333).
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.
        horizontal_flip_prob: Probability of horizontal flip. Default: 0.5.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.RandomResizedCrop(
            input_size,
            scale=scale,
            ratio=ratio,
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomHorizontalFlip(p=horizontal_flip_prob),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    input_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Get validation data transforms.

    Standard ImageNet-style validation preprocessing:
    1. Resize: Resize shorter side to resize_size
    2. CenterCrop: Center crop to input_size × input_size
    3. ToTensor: Convert PIL Image to tensor [0, 1]
    4. Normalize: Standardize with ImageNet mean/std

    Args:
        input_size: Target crop size. Default: 224.
        resize_size: Size to resize the shorter edge. Default: 256.
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    input_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Get test data transforms (same as validation).

    Args:
        input_size: Target crop size. Default: 224.
        resize_size: Size to resize the shorter edge. Default: 256.
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return get_val_transforms(
        input_size=input_size,
        resize_size=resize_size,
        mean=mean,
        std=std,
    )


def get_denormalize(
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Normalize:
    """Get de-normalization transform (for visualization).

    Reverses ImageNet normalization to display images.

    Args:
        mean: Channel-wise mean used during normalization.
        std: Channel-wise standard deviation used during normalization.

    Returns:
        De-normalization Normalize transform.

    Example:
        >>> denorm = get_denormalize()
        >>> img_original = denorm(normalized_tensor)
    """
    inv_mean = tuple(-m / s for m, s in zip(mean, std))
    inv_std = tuple(1.0 / s for s in std)
    return T.Normalize(mean=inv_mean, std=inv_std)


if __name__ == "__main__":
    # Quick test
    print("ImageNet Normalization:")
    print(f"  mean: {IMAGENET_MEAN}")
    print(f"  std:  {IMAGENET_STD}")

    train_tf = get_train_transforms()
    val_tf = get_val_transforms()

    print(f"\nTraining transforms: {train_tf}")
    print(f"\nValidation transforms: {val_tf}")

    import torch
    dummy = torch.randn(3, 224, 224)
    denorm = get_denormalize()
    restored = denorm(dummy)
    print(f"\nDe-normalization preserves shape: {dummy.shape == restored.shape}")
