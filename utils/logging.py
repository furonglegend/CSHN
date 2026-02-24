"""
Logging utilities.

- setup_logger(name, log_dir=None, level=logging.INFO) returns a Python logger.
- MetricLogger wraps TensorBoard SummaryWriter and provides simple add_scalar interface.
  If tensorboard is unavailable, MetricLogger falls back to a no-op writer.

This module does not introduce a hard dependency on tensorboard; it's optional.
"""
import logging
from pathlib import Path
from typing import Optional
import os

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None
    _TB_AVAILABLE = False

class MetricLogger:
    """
    Lightweight wrapper for TensorBoard SummaryWriter with graceful fallback.
    """
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        if _TB_AVAILABLE and log_dir is not None:
            self.writer = SummaryWriter(log_dir)
            self.available = True
        else:
            self.writer = None
            self.available = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None) -> None:
        if self.available:
            self.writer.add_scalar(tag, scalar_value, global_step)
        else:
            # fallback: no-op (we keep behavior silent)
            pass

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int = None) -> None:
        if self.available:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def flush(self) -> None:
        if self.available:
            self.writer.flush()

    def close(self) -> None:
        if self.available:
            self.writer.close()

def setup_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger that prints to stdout and optionally to a file.

    Args:
      name: logger name
      log_dir: optional directory to write a log.txt file
      level: logging level

    Returns:
      logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "log.txt")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # propagate False avoids duplicate logging to root
    logger.propagate = False
    return logger