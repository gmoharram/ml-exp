import numpy as np
from collections import Counter


def create_dataset_summary(datamodule, task_name):
    """Create a summary of dataset statistics.
    
    Args:
        datamodule: Lightning DataModule instance
        task_name: Name of the task (e.g. 'image_classification')
        work_dir: Working directory to save summary
        wandb_logger: Optional WandB logger instance
    """
    # Print basic dataset info
    print("Dataset Summary")
    print("=" * 50)
    
    # Get dataset sizes
    train_size = len(datamodule.train_dataloader().dataset)
    val_dataloaders = datamodule.val_dataloader()
    if isinstance(val_dataloaders, dict):
        val_sizes = {k: len(dl.dataset) for k, dl in val_dataloaders.items()}
    else:
        val_sizes = {"val": len(val_dataloaders.dataset)}
    
    test_loader = datamodule.test_dataloader()
    test_size = len(test_loader.dataset) if test_loader is not None else 0
    
    print(f"\nDataset Sizes:")
    print(f"Training samples: {train_size}")
    for split, size in val_sizes.items():
        print(f"{split.capitalize()} samples: {size}")
    if test_size > 0:
        print(f"Test samples: {test_size}")
    
    # For classification tasks, compute class distribution
    if task_name == "image_classification":
        print("\nClass Distribution:")
        
        def get_labels(dataloader):
            all_labels = []
            for _, y, *_ in dataloader:
                all_labels.extend(y.numpy())
            return np.array(all_labels)
        
        # Training set distribution
        train_labels = get_labels(datamodule.train_dataloader())
        train_dist = Counter(train_labels)
        print("\nTraining Set:")
        for class_idx in range(datamodule.num_classes):
            count = train_dist[class_idx]
            percentage = (count / train_size) * 100
            print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")
        
        # Validation set distribution
        print("\nValidation Set:")
        if isinstance(val_dataloaders, dict):
            for split_name, val_loader in val_dataloaders.items():
                val_labels = get_labels(val_loader)
                val_dist = Counter(val_labels)
                print(f"\n{split_name.capitalize()}:")
                for class_idx in range(datamodule.num_classes):
                    count = val_dist[class_idx]
                    percentage = (count / len(val_labels)) * 100
                    print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")
        else:
            val_labels = get_labels(val_dataloaders)
            val_dist = Counter(val_labels)
            for class_idx in range(datamodule.num_classes):
                count = val_dist[class_idx]
                percentage = (count / len(val_labels)) * 100
                print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")