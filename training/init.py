"""
Training package for SphUnc.

Provides:
- Trainer abstraction
- Callbacks
- Optimizer / scheduler factory
- Training logger
"""

from .trainer import Trainer
from .callbacks import CheckpointCallback, EarlyStoppingCallback, LRSchedulerCallback
from .optim import build_optimizer, build_scheduler
from .logger import TrainingLogger

__all__ = [
    "Trainer",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LRSchedulerCallback",
    "build_optimizer",
    "build_scheduler",
    "TrainingLogger",
]