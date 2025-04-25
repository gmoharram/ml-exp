#!/usr/bin/env python3
import os
import logging
import sys
from pathlib import Path
import random

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    # Lazy import for non basic hydra 
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import LearningRateMonitor

    from ml_exp.models import PLModuleWrapper
    from ml_exp.utils import model_summary, create_dataset_summary, TimerCallback

    # Add cuda device name to config before logging
    if device.type == "cuda":
        cfg.device_name = torch.cuda.get_device_name(0)

    # Get working directory (set by Hydra)
    work_dir = Path(os.getcwd())
    log.info(f"Working directory: {work_dir}")

    # Make things as deterministic as possible for reproducibility if seed is specified
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
    else:
        log.info("No seed specified in config, reproducibility settings disabled")

    try:
        # Initialize Weights & Biases
        if cfg.get("logger", {}).get("project"):
            wandb.init(
                project=cfg.logger.project,
                entity=cfg.logger.entity,
                mode=cfg.logger.mode,
                name=cfg.logger.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=work_dir
            )
            wandb_logger = WandbLogger(
                project=cfg.logger.project,
                entity=cfg.logger.entity,
                name=cfg.logger.name
            )
        else:
            log.info("Weights & Biases is disabled")
            wandb_logger = None

        # Create datamodule
        log.info(f"Creating datamodule: {cfg.data_module.dataset.name}")
        # pop model_name from cfg.pl_module.model since it's not a valid argument for the currently used model _target_ (TODO: change if beneficial)
        OmegaConf.set_struct(cfg, False) # necessary to be able to pop the model_name
        model_name = cfg.pl_module.model.pop("name") # used for model-specific transforms
        accumulate_grad_batches = cfg.data_module.dataset.pop("accumulate_grad_batches")
        OmegaConf.set_struct(cfg, True) # disallow conf modifications again
        datamodule = hydra.utils.instantiate(
            cfg.data_module.dataset, 
            model_name=model_name, 
            augmentations_cfg=cfg.data_module.augmentations,
            seed=cfg.seed  # Pass the seed from config
        )
        
        # Create and log dataset summary
        if cfg.get('create_data_summary', True):
            log.info("Creating dataset summary")
            dataset_summary_path = work_dir / "dataset_summary.txt"
            with open(dataset_summary_path, "w") as f:
                sys.stdout = f
                create_dataset_summary(datamodule, cfg.task_name)
                sys.stdout = sys.__stdout__
            if wandb_logger is not None:
                wandb.save(str(dataset_summary_path), base_path=str(work_dir))

        # Create Lightning module
        log.info("Creating model")
        # Update scheduler T_max if not set and using cosine scheduler
        scheduler_interval = 'step' # will be used for LearningRateMonitor either way
        if cfg.pl_module.get("scheduler") is not None:
            scheduler_interval = cfg.pl_module.scheduler.interval
            # convert milestones to num steps if using step scheduler
            if getattr(cfg.pl_module.scheduler, "milestones", None):
                cfg.pl_module.scheduler.milestones = [int(len(datamodule.train_dataloader()) * milestone / accumulate_grad_batches) for milestone in cfg.pl_module.scheduler.milestones]
            for scheduler in cfg.pl_module.scheduler.schedulers:    
                if scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR":
                    if scheduler.T_max is None:
                        scheduler.T_max = cfg.trainer.trainer.max_epochs
                    if scheduler_interval == "step":
                        scheduler.T_max = int(len(datamodule.train_dataloader()) * scheduler.T_max / accumulate_grad_batches)
                elif scheduler._target_ == "torch.optim.lr_scheduler.LinearLR":
                    if scheduler.total_iters is None:
                        raise ValueError("total_iters in epochs must be set for linear scheduler")
                    if scheduler_interval == "step":
                        scheduler.total_iters = int(
                            len(datamodule.train_dataloader()) * scheduler.total_iters / accumulate_grad_batches
                        )
                elif (scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"):
                    # transform T_0 value to num steps if using step scheduler
                    if scheduler_interval == "step":
                        scheduler.T_0 = scheduler.T_0 * len(datamodule.train_dataloader())

        # Log transfer learning info if enabled
        if cfg.pl_module.transfer.enabled:
            log.info(f"Transfer learning enabled. Loading from: {cfg.pl_module.transfer.checkpoint_path}")
            log.info("Freezing backbone.") if cfg.pl_module.transfer.freeze_backbone else log.info("Not freezing backbone.")
            log.info("Reinitializing head.") if cfg.pl_module.transfer.reinit_head else log.info("Not reinitializing head.")

        # Load model
        model = PLModuleWrapper(
            task_name=cfg.task_name,
            model_config=dict(cfg.pl_module.model),
            criterion_config=dict(cfg.pl_module.criterion),
            optimizer_config=dict(cfg.pl_module.optimizer),
            transfer_config=dict(cfg.pl_module.transfer),
            scheduler_config=dict(cfg.pl_module.scheduler) if cfg.pl_module.get("scheduler") else None,
        )

        # Log model summary
        if cfg.get('create_model_summary', True):
            log.info("Logging model summary")
            # example_input = datamodule.val_dataloader().dataset[0][0] # Get example input from datamodule, val due to shuffle=False
            input_shape = (1, cfg.data_module.dataset.in_channels, cfg.data_module.dataset.image_size, cfg.data_module.dataset.image_size) # Create dummy input tensor
            dummy_input = torch.ones(input_shape)
            model_summary_path = work_dir / "model_summary.txt"
            with open(model_summary_path, "w") as f:
                # Redirect stdout to file
                sys.stdout = f
                total_params, total_param_bytes, total_output_size, total_flops = model_summary(model.to(device), dummy_input.to(device))
                # Restore stdout
                sys.stdout = sys.__stdout__
            if wandb_logger is not None:
                wandb.save(str(model_summary_path), base_path=str(work_dir))
                # Log model statistics to wandb
                wandb.log({
                    "model/total_params_millions": total_params / 1e6,
                    "model/total_param_bytes_mb": total_param_bytes / 1e6,
                    "model/total_output_size_mb": total_output_size / 1e6,
                    "model/total_estimated_memory_usage_mb": (total_param_bytes + total_output_size) / 1e6,
                    "model/total_flops_giga": total_flops / 1e9,
                })
        # Create trainer
        log.info("Creating trainer")

        # Setup checkpointing
        checkpoint_callbacks = [
            LearningRateMonitor(logging_interval=scheduler_interval),
            TimerCallback(log_to_file=True)
        ]
        for checkpoint_name, checkpoint_cfg in cfg.trainer.checkpoint.items():
            checkpoint_cfg.dirpath = work_dir / "checkpoints"
            checkpoint_callbacks.append(ModelCheckpoint(**checkpoint_cfg))

        OmegaConf.set_struct(cfg, False) # necessary to be able to pop values from cfg
        precision = cfg.trainer.trainer.pop("precision")
        if precision == "32-matmul":
            torch.set_float32_matmul_precision("medium")
            precision = "32-true"
        OmegaConf.set_struct(cfg, True) # disallow conf modifications again
        
        trainer = pl.Trainer(
            **cfg.trainer.trainer,
            logger=wandb_logger,
            callbacks=checkpoint_callbacks,
            accumulate_grad_batches=accumulate_grad_batches,
            precision=precision
        )
        
        # Train model
        log.info("Starting training.")
        trainer.fit(model, datamodule=datamodule)
        
        # Log best model information
        log.info("Training completed.")

    except Exception as e:

        log.error(f"Training failed with error: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        raise e

    
    # Test if test set is available
    if hasattr(datamodule, "test_dataloader") and datamodule.test_dataloader() is not None:
        log.info("Skipping testing.")
        # log.info("Starting testing")
        # trainer.test(model, datamodule=datamodule)
    
    # Close wandb run
    if wandb_logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
