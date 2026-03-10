"""Image augmentation pipelines for training and validation."""

from __future__ import annotations

from torchvision import transforms

# ImageNet statistics — reasonable default for ISIC images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int) -> transforms.Compose:
    """Return strong augmentation pipeline for training.

    Includes random flips, rotation, affine, color jitter, random erasing,
    and normalization.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10
            ),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def get_val_transforms(image_size: int) -> transforms.Compose:
    """Return deterministic pipeline for validation and test inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
