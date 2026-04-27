"""
================================================================================
SEAT BELT DATASET
================================================================================

PyTorch Dataset for seat belt binary classification.

Assumes:
    - Images are already upper-body ROI crops
    - Class-folder structure (ImageFolder-style)
    - Binary labels: 0 = no seat belt, 1 = wearing seat belt

Directory Structure:
    train/
    ├── PassengerWithSeatBelt_aug/
    │   ├── image001.jpg
    │   └── ...
    └── PassengerWithoutSeatBelt_aug/
        ├── image001.jpg
        └── ...

================================================================================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable
import random


class SeatBeltDataset(Dataset):
    """
    Dataset for seat belt binary classification.
    
    Loads images from class-folder structure and returns:
        - image: (3, H, W) tensor
        - label: 0 (no belt) or 1 (with belt)
    """
    
    # Class name to label mapping
    CLASS_MAPPING = {
        'PassengerWithSeatBelt_aug': 1,
        'PassengerWithSeatBelt': 1,
        'WithSeatBelt': 1,
        'with_belt': 1,
        'positive': 1,
        '1': 1,
        'PassengerWithoutSeatBelt_aug': 0,
        'PassengerWithoutSeatBelt': 0,
        'WithoutSeatBelt': 0,
        'without_belt': 0,
        'negative': 0,
        '0': 0,
    }
    
    def __init__(
        self,
        root_dirs: List[str],
        transform: Optional[Callable] = None,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """
        Initialize dataset.
        
        Args:
            root_dirs: List of root directories containing class folders
            transform: Optional torchvision transforms
            image_extensions: Valid image file extensions
        """
        self.transform = transform
        self.image_extensions = image_extensions
        self.samples: List[Tuple[str, int]] = []
        
        # Collect all samples from all root directories
        for root_dir in root_dirs:
            self._scan_directory(root_dir)
        
        # Shuffle samples
        random.shuffle(self.samples)
        
        # Compute class distribution
        self.class_counts = self._compute_class_distribution()
    
    def _scan_directory(self, root_dir: str):
        """Scan directory for class folders and images."""
        root_path = Path(root_dir)
        
        if not root_path.exists():
            print(f"Warning: Directory does not exist: {root_dir}")
            return
        
        # Check if root_dir itself is a class folder
        folder_name = root_path.name
        if folder_name in self.CLASS_MAPPING:
            label = self.CLASS_MAPPING[folder_name]
            self._add_images_from_folder(root_path, label)
        else:
            # Scan subdirectories as class folders
            for class_dir in root_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    if class_name in self.CLASS_MAPPING:
                        label = self.CLASS_MAPPING[class_name]
                        self._add_images_from_folder(class_dir, label)
    
    def _add_images_from_folder(self, folder: Path, label: int):
        """Add all images from folder with given label."""
        for ext in self.image_extensions:
            for img_path in folder.glob(f'*{ext}'):
                self.samples.append((str(img_path), label))
            for img_path in folder.glob(f'*{ext.upper()}'):
                self.samples.append((str(img_path), label))
    
    def _compute_class_distribution(self) -> Dict[int, int]:
        """Compute number of samples per class."""
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with:
                'image': (3, H, W) tensor
                'label': int (0 or 1)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            import torchvision.transforms as T
            image = T.ToTensor()(image)
        
        return {
            'image': image,
            'label': label
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset.
        
        Returns:
            weights: Tensor of shape (2,) with inverse frequency weights
        """
        total = len(self.samples)
        weights = torch.tensor([
            total / (2 * self.class_counts[0]) if self.class_counts[0] > 0 else 1.0,
            total / (2 * self.class_counts[1]) if self.class_counts[1] > 0 else 1.0
        ])
        return weights


def create_dataloaders(
    train_dirs: List[str],
    val_dirs: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dirs: List of training data directories
        val_dirs: List of validation data directories
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        augment_train: Whether to apply augmentation to training data
        
    Returns:
        train_loader, val_loader
    """
    import torchvision.transforms as T
    
    # ImageNet normalization
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms (with augmentation)
    if augment_train:
        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize
        ])
    
    # Validation transforms (no augmentation)
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize
    ])
    
    # Create datasets
    train_dataset = SeatBeltDataset(train_dirs, transform=train_transform)
    val_dataset = SeatBeltDataset(val_dirs, transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"  - With belt: {train_dataset.class_counts[1]}")
    print(f"  - Without belt: {train_dataset.class_counts[0]}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"  - With belt: {val_dataset.class_counts[1]}")
    print(f"  - Without belt: {val_dataset.class_counts[0]}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inspect a seat belt dataset split')
    parser.add_argument('dirs', nargs='+', help='Dataset split root(s) or class folder(s)')
    args = parser.parse_args()

    dataset = SeatBeltDataset(args.dirs)
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution: {dataset.class_counts}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample label: {sample['label']}")
