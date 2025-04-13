import torch

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of steps for linear warmup.
            total_steps: Total number of steps (warmup + cosine annealing).
            eta_min: Minimum learning rate.
            last_epoch: The index of the last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.cosine_steps = total_steps - warmup_steps
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [(base_lr * (self.last_epoch + 1) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / self.cosine_steps
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + torch.cos(torch.tensor(torch.pi * progress))) / 2
                for base_lr in self.base_lrs
            ]
