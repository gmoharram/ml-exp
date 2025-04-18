from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import medmnist
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset


class DataModuleWrapper(pl.LightningDataModule):
    """Data module for handling datasets in a PyTorch Lightning compatible way."""
    
    def __init__(
        self,
        name: str,
        root_dir: str,
        image_size: int,
        in_channels: int,
        num_classes: int,
        batch_size: int,
        model_name: str,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        download: bool = False,
        trainset_fraction: Optional[float] = None,
        valset_fraction: Optional[float] = None,
        transforms_cfg: Dict[str, Any] = {},
        augmentations_cfg: Dict[str, Any] = {},
        seed: int = 42,
    ):
        """
        Initialize the data module.
        
        Args:
            name: Name of the dataset (e.g., 'camelyon17', 'breastmnist')
            root_dir: Root directory for dataset storage
            image_size: Size of the images
            in_channels: Number of channels in the images
            num_classes: Number of classes in the dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            persistent_workers: Whether to use persistent workers for dataloaders
            pin_memory: Whether to pin memory for dataloaders
            download: Whether to download the dataset
            trainset_fraction: Fraction of training set to use (between 0 and 1, None means use entire dataset)
            valset_fraction: Fraction of validation sets to use (between 0 and 1, None means use entire datasets)
            augmentations_cfg: Config for the augmentations
            transforms_cfg: Config for the transforms
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize datasets to None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.ood_val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Set up transforms & load dataset
        self.transform_fn = self._get_transforms(transforms_cfg)
        self.train_transform_fn = self._get_augmentations(augmentations_cfg, transforms_fn=self.transform_fn)
        self.setup()
        
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the datasets. 
        'stage' is either None,'fit', 'validate', 'test', or 'predict'. Not used yet.
        This method is called on every GPU.
        """
        # Get dataset splits depending on the dataset
        (
            self.train_dataset,
            self.val_dataset,
            self.ood_val_dataset,
            self.test_dataset
        ) = self._get_splits()
            
        # Subset training data if trainset_fraction is specified
        self.train_dataset = self._subset_dataset(self.train_dataset, self.hparams.trainset_fraction, shuffle=True)

        # Subset validation datasets if valset_fraction is specified
        self.val_dataset = self._subset_dataset(self.val_dataset, self.hparams.valset_fraction, shuffle=False)
        if self.ood_val_dataset is not None:
            self.ood_val_dataset = self._subset_dataset(self.ood_val_dataset, self.hparams.valset_fraction, shuffle=False)
            
    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        # Create a generator for reproducible shuffling only if seed is specified
        generator = None
        if self.hparams.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.hparams.seed)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            generator=generator  # Will be None if no seed specified
        )

    def val_dataloader(self) -> Dict[str, DataLoader]:
        """Create the validation dataloaders."""
        dataloaders = {
            'val': DataLoader(
                self.val_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                persistent_workers=self.hparams.persistent_workers,
                pin_memory=self.hparams.pin_memory
            )
        }
        
        if self.ood_val_dataset is not None:
            dataloaders['ood_val'] = DataLoader(
                self.ood_val_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                persistent_workers=self.hparams.persistent_workers,
                pin_memory=self.hparams.pin_memory
            )
            
        return dataloaders

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create the test dataloader."""
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=self.hparams.pin_memory
        )
        
    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return self.hparams.num_classes
            
    @property
    def in_channels(self) -> int:
        """Get the number of input channels."""
        return self.hparams.in_channels
    
    def _get_transforms(self, config: Dict[str, Any]) -> transforms.Compose:
        """Get the dataset (& model) specific transforms."""
        transforms_list = config.get("defaults", [])
        if self.hparams.model_name in config:
            transforms_list += config[self.hparams.model_name]
        return transforms.Compose(transforms_list)
    
    def _get_augmentations(self, config: Dict[str, Any], transforms_fn: transforms.Compose) -> transforms.Compose:
        """Get the trainset augmentations."""
        augmentations_list = config.get("augmentations_list", []) 
        augmentations_list.append(transforms_fn)
        if self.hparams.model_name in config:
            augmentations_list += config[self.hparams.model_name]
        return transforms.Compose(augmentations_list)

    def _subset_dataset(self, dataset: Dataset, fraction: Optional[float], shuffle: bool = True) -> Dataset:
        """Helper function to subset a dataset by a given fraction."""
        if fraction is not None:
            if not 0 < fraction <= 1:
                raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
            num_samples = len(dataset)
            num_samples_to_use = int(num_samples * fraction)
            if shuffle:
                # Use generator only if seed is specified
                generator = None
                if self.hparams.seed is not None:
                    generator = torch.Generator()
                    generator.manual_seed(self.hparams.seed)
                indices = torch.randperm(num_samples, generator=generator)[:num_samples_to_use]
            else:
                indices = torch.arange(num_samples_to_use)
            return Subset(dataset, indices)
        return dataset

    def _get_splits(self) -> Tuple[Dataset, Dataset, Optional[Dataset], Optional[Dataset]]:
        """Get train, validation, ood validation and test splits for the dataset.
        
        Returns:
            Tuple of (train_dataset, val_dataset, ood_val_dataset, test_dataset)
            where ood_val_dataset and test_dataset may be None if not available
        """
        if self.hparams.name == "camelyon17":
            full_dataset = Camelyon17Dataset(root_dir=Path(self.hparams.root_dir), download=self.hparams.download)
            train_dataset = full_dataset.get_subset('train', transform=self.train_transform_fn)
            val_dataset = full_dataset.get_subset('id_val', transform=self.transform_fn)
            ood_val_dataset = full_dataset.get_subset('val', transform=self.transform_fn)
            test_dataset = full_dataset.get_subset('test', transform=self.transform_fn)
            
        elif "mnist" in self.hparams.name:  # MedMNIST datasets
            info = medmnist.INFO[self.hparams.name]
            
            # Load datasets
            DataClass = getattr(medmnist, info["python_class"])
            train_dataset = DataClass(
                split='train',
                transform=self.train_transform_fn,
                download=self.hparams.download,
                size=self.hparams.image_size,
                root=self.hparams.root_dir,
            )
            val_dataset = DataClass(
                split='val',
                transform=self.transform_fn,
                download=self.hparams.download,
                size=self.hparams.image_size,
                root=self.hparams.root_dir,
            )
            test_dataset = DataClass(
                split='test',
                transform=self.transform_fn,
                download=self.hparams.download,
                size=self.hparams.image_size,
                root=self.hparams.root_dir,
            )
            ood_val_dataset = None
            
            # Convert labels if needed
            for dataset in [train_dataset, val_dataset, test_dataset]:
                if dataset.labels.shape[1] > 1:  # convert one-hot to index
                    dataset.labels = dataset.labels.argmax(axis=1)
                else:
                    dataset.labels = dataset.labels.flatten()
                    
        else:
            raise ValueError(f"Dataset {self.hparams.name} not supported")
            
        return train_dataset, val_dataset, ood_val_dataset, test_dataset