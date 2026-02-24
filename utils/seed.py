"""
Set deterministic seeds for reproducibility across numpy, random, torch.

Note: full determinism in deep learning may require environment-specific flags.
This function attempts to set seeds for common libraries.
"""
import random
import os
from typing import Optional

def set_seed(seed: Optional[int]) -> None:
    """
    Set Python, NumPy, and PyTorch seeds.

    Args:
      seed: integer seed. If None, does nothing.
    """
    if seed is None:
        return

    try:
        import numpy as _np
        _np.random.seed(int(seed))
    except Exception:
        pass

    try:
        import torch as _torch
        _torch.manual_seed(int(seed))
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(int(seed))
            # recommended flags for deterministic behavior (may reduce performance)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    random.seed(int(seed))
    # set PYTHONHASHSEED for consistent hashing
    os.environ['PYTHONHASHSEED'] = str(seed)