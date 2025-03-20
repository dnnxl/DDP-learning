from torch.utils.data import Dataset
from pathlib import Path
from functools import partial
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

def exists(x):
    return x is not None

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    """
    A custom PyTorch dataset class for loading images from a folder.

    Attributes:
        folder (str): The directory containing the images.
        image_size (int or tuple): The desired image size for resizing.
        exts (list, optional): List of allowed image file extensions (default: ['jpg', 'jpeg', 'png', 'tiff']).
        augment_horizontal_flip (bool, optional): If True, applies random horizontal flipping (default: False).
        convert_image_to (str, optional): Color mode conversion (e.g., 'RGB', 'L' for grayscale) (default: None).

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(index): Loads and transforms an image at the specified index.
    """

    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        """
        Initializes the dataset by loading image paths and defining transformations.

        Args:
            folder (str): Path to the directory containing images.
            image_size (int or tuple): Target image size for resizing.
            exts (list, optional): Allowed image file extensions.
            augment_horizontal_flip (bool, optional): Whether to apply random horizontal flip.
            convert_image_to (str, optional): Color mode conversion (e.g., 'RGB').
        """
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # Collect all image file paths with the given extensions
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # Define a conversion function if color mode conversion is needed
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # Define image transformation pipeline
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),  # Convert color mode if needed
            T.Resize(image_size),  # Resize image to target size
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),  # Apply random flip if enabled
            T.CenterCrop(image_size),  # Crop to target size
            T.ToTensor()  # Convert image to tensor
        ])

    def __len__(self):
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, index):
        """
        Loads and applies transformations to an image.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        path = self.paths[index]  # Get image path
        img = Image.open(path)  # Load image
        return self.transform(img)  # Apply transformations
