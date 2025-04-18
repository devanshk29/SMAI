import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class OCRDataset(Dataset):
    def __init__(self, image_dir, char_to_idx, transform=None, max_length=30):
        """
        Initialize the OCR dataset.

        Args:
            image_dir (str): Path to the directory containing images.
            char_to_idx (dict): Mapping of characters to indices.
            transform (callable, optional): Transformation to apply to the images.
            max_length (int): Maximum length of encoded labels (for padding).
        """
        self.image_dir = image_dir
        self.image_files = [file for file in os.listdir(image_dir) if file.endswith(".png")]
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding encoded label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed image, padded label tensor)
        """
        # Get the filename and label
        image_name = self.image_files[idx]
        label = os.path.splitext(image_name)[0]  # Extract label by removing the file extension

        # Encode the label using the character-to-index mapping
        label_encoded = [self.char_to_idx[char] for char in label]

        # Load and transform the image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Create a padded label tensor
        label_tensor = torch.full((self.max_length,), -1, dtype=torch.long)  # Padding token (-1)
        label_tensor[:len(label_encoded)] = torch.tensor(label_encoded)

        return image, label_tensor
