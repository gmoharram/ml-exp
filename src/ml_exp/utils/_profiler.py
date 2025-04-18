import os
import psutil

import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.profiler import profile
from torchinfo import summary


def model_summary(model, input: torch.Tensor, depth=3, exec_times=False, recount_params=True):
    """Profile your pytorch model with respect to parameters, FLOPS, and execution times.
    Args:
        - model: Basic pytorch module
        - input: Input tensor - should be on same device as model

    Returns:
        - total_params: All parameters of the model
        - total_flops: FLOPS required for given input shape
    """
    print("\n################# PARAMETERS #################")
    results = summary(model, input_size=input.size(), depth=depth)
    if recount_params:
        results.total_params = sum(p.numel() for p in model.parameters())
        results.total_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(results)

    # Get FLOPs and execution times
    print("\n################# FLOPS #################")
    print("==========================================")
    with profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with FlopCounterMode(depth=depth) as flop_profiler:
            model(input)
    print("==========================================")

    if exec_times:
        print("\n################# EXECUTION TIMES #################")
        print(prof.key_averages().table())
    return results.total_params, results.total_param_bytes, results.total_output_bytes, flop_profiler.get_total_flops()

def log_memory_usage(device=None):
    # Get current process
    process = psutil.Process(os.getpid())

    # Get RAM usage in MB
    ram_usage = process.memory_info().rss / (1024**2)

    # Get disk usage in MB
    disk_usage = psutil.disk_usage("/").used / (1024**2)

    # Log the memory usage
    print(f"CPU RAM Usage: {ram_usage:.2f} MB")
    print(f"CPU Disk Usage: {disk_usage:.2f} MB")

    # Get GPU memory usage
    if device is not None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(device, torch.device):
            gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)
            print(f"GPU RAM Allocated: {gpu_memory:.2f} MB")
            gpu_memory = torch.cuda.memory_reserved(device) / (1024**2)
            print(f"GPU RAM Reserved: {gpu_memory:.2f} MB")
