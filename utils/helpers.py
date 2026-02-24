"""
Small helper utilities.

- topk_indices(scores, k) -> indices of top-k values (ties preserved by order)
- ensure_dir(path) -> create directory
"""
from pathlib import Path
import numpy as np
from typing import List, Sequence

def topk_indices(scores: Sequence[float], k: int) -> List[int]:
    """
    Return indices of top-k values in 'scores' (largest first).
    If k >= len(scores) returns all indices sorted.
    """
    arr = np.asarray(scores)
    if k <= 0:
        return []
    k = min(k, arr.size)
    # use argpartition for efficiency
    idx = np.argpartition(-arr, k-1)[:k]
    sorted_idx = idx[np.argsort(-arr[idx])]
    return sorted_idx.tolist()

def ensure_dir(path: str) -> None:
    """
    Create a directory (including parents) if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)