import torch
from typing import Optional


def get_device(overwrite: Optional[bool] = None, device: str = 'cuda:0', fallback_device: str = 'cpu') -> torch.device:
    use_cuda = torch.cuda.is_available() if overwrite is None else overwrite
    return torch.device(device if use_cuda else fallback_device)
