from typing import Any, Dict, Optional
import hydra
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, AUROC, Precision, Recall, F1Score
import torch
from collections import Counter

class PLModuleWrapper(pl.LightningModule):
    """Base Lightning model class for all models."""
    
    def __init__(
        self,
        task_name: str,
        model_config: Dict[str, Any],
        criterion_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        transfer_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model.
        
        Args:
            model_config: Configuration for the model with structure:
                {
                    "_target_": "path.to.ModelClass",
                    "param1": value1,
                    ...
                }
            criterion_config: Configuration for the loss function with structure:
                {
                    "_target_": "torch.nn.CrossEntropyLoss",
                    "param1": value1,
                    ...
                }
            optimizer_config: Configuration for the optimizer with structure:
                {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                    ...
                }
            scheduler_config: Optional configuration for learning rate scheduler with structure:
                {
                    "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "T_max": int,
                    "eta_min": float,
                    "monitor": str,
                    "interval": str,
                    "frequency": int
                }
            transfer_config: Configuration for transfer learning with structure:
                {
                    "enabled": bool,
                    "checkpoint_path": str,
                    "freeze_backbone": bool,
                    "reinit_head": bool
                }
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Instantiate model
        self.model = hydra.utils.instantiate(model_config)

        if self.hparams.transfer_config.get("enabled", False):
            print("Don't forget to call setup_transfer_learning_only() when loading the model without a trainer.")

    def setup(self, stage: Optional[str] = None):
        """Setup is called on every GPU separately and offers lazy initialization."""
        num_classes = self.hparams.model_config.get("num_classes", None)
        self._setup_metrics(num_classes)
        self._setup_loss(num_classes)

        if self.hparams.transfer_config.get("enabled", False):
            self._setup_transfer_learning(self.hparams.transfer_config, num_classes)
    
    def setup_transfer_learning_only(self):
        """Setup transfer learning only when pl_module is loaded without trainer."""
        num_classes = self.hparams.model_config.get("num_classes", None)
        self._setup_transfer_learning(self.hparams.transfer_config, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _compute_loss(self, y_hat, y):
        """Compute the loss.
        
        Args:
            batch: A tuple containing (x, y) or (x, y, metadata)
                x: Input data
                y: Target labels
                metadata: Optional additional information
        """
        return self.criterion(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Unpack batch with optional metadata
        x, y, *metadata = batch
        y_hat = self(x)

        # Compute loss
        loss = self._compute_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Calculate detailed metrics
        self.train_metrics.update(y_hat, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        # The dataloader_idx will be 0 for the ID validation and 1 for the OOD validation"
        prefix = "val" if dataloader_idx == 0 else "ood_val"

        # Unpack batch with optional metadata
        x, y, *metadata = batch
        y_hat = self(x)

        # Compute loss
        loss = self._compute_loss(y_hat, y)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        # Update detailed metrics, 
        self.val_metrics[dataloader_idx].update(y_hat, y)

        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        # Unpack batch with optional metadata
        x, y, *metadata = batch
        y_hat = self(x)

        # Compute loss
        loss = self._compute_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        # Update detailed metrics
        self.test_metrics.update(y_hat, y)

        return loss

    def on_validation_epoch_end(self):
        # probably can do the same as training_step with on_step=False and let pl handle reset. But this works for sure.
        for dataloader_idx in range(len(self.val_metrics)):
            metric_epoch_output = self.val_metrics[dataloader_idx].compute()
            self.log_dict(metric_epoch_output)
            self.val_metrics[dataloader_idx].reset()
        
    def on_test_epoch_end(self):
        metric_epoch_output = self.test_metrics.compute()
        self.log_dict(metric_epoch_output)
        self.test_metrics.reset()
        
        

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Initialize optimizer with parameters
        optimizer = hydra.utils.instantiate(self.hparams.optimizer_config, params=self.parameters())
        
        if not self.hparams.scheduler_config:
            return optimizer
            
        # Initialize scheduler
        scheduler_config = dict(self.hparams.scheduler_config)
        monitor = scheduler_config.pop("monitor", None)
        interval = scheduler_config.pop("interval", "step")
        frequency = scheduler_config.pop("frequency", 1)
        milestones = scheduler_config.pop("milestones", [])

        # Instantiate each scheduler
        schedulers = []
        for scheduler_cfg in scheduler_config['schedulers']:
            scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
            schedulers.append(scheduler)
        
        # Create SequentialLR with the schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=milestones)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "name": "learning_rate", # same logging name for all optimizers for wandb graph
                "monitor": monitor,
                "interval": interval,
                "frequency": frequency,
            }
        }

    def _setup_loss(self, num_classes: Optional[int] = None):
        """Initialize loss function with optional class weights for CrossEntropyLoss."""
        if self.hparams.criterion_config.pop("weighted", False):
            # Get training data distribution
            train_labels = []
            for _, y, *_ in self.trainer.datamodule.train_dataloader():
                train_labels.extend(y.numpy())
            train_dist = Counter(train_labels)
            
            # Calculate inverse frequency weights
            total_samples = len(train_labels)
            weights = []
            for class_idx in range(num_classes):
                count = train_dist[class_idx]
                if count == 0:
                    weights.append(1.0)  # Handle class with no samples
                else:
                    weights.append(total_samples / (num_classes * count))
            
            # Convert to tensor and normalize
            class_weights = torch.tensor(weights, dtype=torch.float32)
            class_weights = class_weights / class_weights.sum() * num_classes
            
            # Add weights to criterion config
            self.hparams.criterion_config["weight"] = class_weights.to(self.device)
            
        # Instantiate criterion with potentially updated config
        self.criterion = hydra.utils.instantiate(self.hparams.criterion_config)

    def _setup_transfer_learning(self, transfer_config: Dict[str, Any], target_num_classes: int):
        """Setup transfer learning by loading weights and optionally freezing layers."""
        if not transfer_config.get("checkpoint_path"):
            raise ValueError("checkpoint_path must be provided when transfer learning is enabled")
            
        # Load checkpoint
        checkpoint = torch.load(transfer_config["checkpoint_path"], map_location=self.device, weights_only=False)
        # Handle different checkpoint formats:
        # 1. Lightning checkpoint: {'state_dict': {'model.layer1.weight': tensor, ...}}
        # 2. Raw state dict: {'layer1.weight': tensor, ...}
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                # Lightning checkpoint - extract model weights and remove 'model.' prefix
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() 
                            if k.startswith("model.")}
            elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                # Raw state dict saved directly from model (not Lightning) - use as is
                state_dict = checkpoint
            else:
                raise ValueError(
                    "Checkpoint format not recognized. Expected either a Lightning checkpoint "
                    "with 'state_dict' key or a raw PyTorch state dict."
                )
        else:
            raise ValueError(
                f"Expected checkpoint to be a dict, got {type(checkpoint)}. "
                "The checkpoint should either be a Lightning checkpoint or a raw state dict."
            )
        
        # Switch out the final layer for the target number of classes if necessary
        if self.hparams.task_name == "image_classification":
            # Check if number of classes is different
            source_num_classes = None
            name, param = list(state_dict.items())[-1]
            if "classifier" in name or "head" in name or "fc" in name:
                # Weight matrix or bias of final layer
                source_num_classes = param.shape[0]
            else:
                raise ValueError(f"Expected final layer to be a classifier, head or fc, got {name}")

            
            if not transfer_config.get("reinit_head", True):
                if source_num_classes != target_num_classes:
                    raise ValueError(
                        f"Number of classes mismatch (source: {source_num_classes}, target: {target_num_classes}) "
                        "and reinit_head is False"
                    )
            else:
                # Remove classification head from state dict
                last_name = list(state_dict.keys())[-1]
                sec_last_name = None
                if 'bias' in last_name:
                    sec_last_name = list(state_dict.keys())[-2]
                    assert ('weight' in sec_last_name) and ('classifier' in sec_last_name or 'head' in sec_last_name or 'fc' in sec_last_name), f"Expected second to last layer to be a weight matrix of a classifier, head or fc layer, instead got {sec_last_name}"
                    state_dict.pop(sec_last_name)
                state_dict.pop(last_name)

        # TODO: Handle cases where in_channels is different
                        
        # Load pretrained weights into model
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading weights: {unexpected_keys}")
        
        # Freeze backbone if requested
        if transfer_config.get("freeze_backbone", False):
            for name, param in self.model.named_parameters():
                if name not in [sec_last_name, last_name]:
                    param.requires_grad = False
            print(f"Backbone frozen frozen.")
    
        print("Transfer learning setup complete.")

    def _setup_metrics(self, num_classes: Optional[int] = None):
        """Initialize train, validation and test metrics."""
        self.train_metrics = self._get_train_metrics(self.hparams.task_name, num_classes).to(self.device)
        
        val_dataloaders = self.trainer.datamodule.val_dataloader()
        num_val_dataloaders = len(val_dataloaders) if isinstance(val_dataloaders, dict) else 1
        prefixes = list(val_dataloaders.keys()) if isinstance(val_dataloaders, dict) else ["val"]
        self.val_metrics = torch.nn.ModuleList([
            self._get_val_metrics(self.hparams.task_name, num_classes, prefix=f"{prefixes[i]}_").to(self.device) 
            for i in range(num_val_dataloaders)
        ])
        
        self.test_metrics = self._get_val_metrics(self.hparams.task_name, num_classes, prefix="test_").to(self.device)

    def _get_train_metrics(self, task_name, num_classes = None):
        """Get train metrics."""
        if task_name == "image_classification":
            if num_classes is None:
                raise ValueError("Config dataset.num_classes must be provided for classification tasks.")
            metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=num_classes, average="micro"),
                "auroc_weighted": AUROC(task="multiclass", num_classes=num_classes, average="weighted"),
                "auroc_macro": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
            })
        else:
            raise ValueError(f"Task name {task_name} not supported for gettingtraining metrics.")

        return metrics.clone(prefix="train_")

    def _get_val_metrics(self, task_name, num_classes = None, prefix = "val_"):
        """Get validation metrics."""
        if task_name == "image_classification":
            if num_classes is None:
                raise ValueError("Config dataset.num_classes must be provided for classification tasks.")
            metrics = MetricCollection(    
                {
                    "acc": Accuracy(task="multiclass", num_classes=num_classes, average="micro"),
                    "auroc_weighted": AUROC(task="multiclass", num_classes=num_classes, average="weighted"),
                    "auroc_macro": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
                    "f1_micro": F1Score(task="multiclass", num_classes=num_classes, average="micro"),
                    "f1_weighted": F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
                    "f1_macro": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                }
            )
        else:
            raise ValueError(f"Task name {task_name} not supported for getting validation metrics.")
        
        return metrics.clone(prefix=prefix)