import time
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
from pathlib import Path


class TimerCallback(Callback):
    """
    Callback to log training time metrics to wandb.
    
    Logs:
    - Time per epoch
    - Iterations per second
    """
    
    def __init__(self, log_to_file=True):
        """
        Args:
            log_to_file: Whether to write timing info to a file (in addition to wandb)
        """
        super().__init__()
        self.log_to_file = log_to_file
        self.epoch_start_time = None
        self.epoch_end_times = []
        self.epoch_durations = []
        self.batches_per_epoch = None
        self.timing_file = None
    
    def on_fit_start(self, trainer, pl_module):
        """Initialize timer file when training starts."""
        if self.log_to_file:
            # Get current working directory (where Hydra is running)
            work_dir = Path(os.getcwd())
            self.timing_file = work_dir / "training_times.txt"
            # Write header
            with open(self.timing_file, "w") as f:
                f.write("epoch,duration_seconds,iterations_per_second\n")
            # Upload to wandb
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                wandb.save(str(self.timing_file), base_path=str(work_dir))
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Start timing at the beginning of each epoch."""
        self.epoch_start_time = time.time()
        if self.batches_per_epoch is None:
            # Store the number of batches per epoch for it/s calculation
            self.batches_per_epoch = len(trainer.train_dataloader)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log timing metrics at the end of each epoch."""
        if self.epoch_start_time is None:
            return
            
        # Calculate timing metrics
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_durations.append(epoch_duration)
        self.epoch_end_times.append(epoch_end_time)
        
        # Calculate iterations per second
        iterations_per_second = self.batches_per_epoch / epoch_duration
        
        # Log to wandb
        trainer.logger.log_metrics({
            "time/epoch_duration": epoch_duration,
            "time/iterations_per_second": iterations_per_second
        }, step=trainer.current_epoch)
        
        # Write to file
        if self.log_to_file and self.timing_file:
            with open(self.timing_file, "a") as f:
                f.write(f"{trainer.current_epoch},{epoch_duration:.4f},{iterations_per_second:.4f}\n")
                
        # Reset timer for next epoch
        self.epoch_start_time = None
    
    def on_fit_end(self, trainer, pl_module):
        """Log overall training time statistics."""
        if not self.epoch_durations:
            return
            
        # Calculate average metrics
        avg_epoch_duration = sum(self.epoch_durations) / len(self.epoch_durations)
        avg_iterations_per_second = self.batches_per_epoch / avg_epoch_duration
        total_training_time = time.time() - self.epoch_end_times[0] + self.epoch_durations[0]
        
        # Log to wandb
        trainer.logger.log_metrics({
            "time/avg_epoch_duration": avg_epoch_duration,
            "time/avg_iterations_per_second": avg_iterations_per_second,
            "time/total_training_time": total_training_time
        })
        
        # Append to file
        if self.log_to_file and self.timing_file:
            with open(self.timing_file, "a") as f:
                f.write(f"\nAverage epoch duration: {avg_epoch_duration:.4f} seconds\n")
                f.write(f"Average iterations per second: {avg_iterations_per_second:.4f}\n")
                f.write(f"Total training time: {total_training_time:.4f} seconds\n")
                
            # Update the file in wandb
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                wandb.save(str(self.timing_file), base_path=str(Path(os.getcwd()))) 