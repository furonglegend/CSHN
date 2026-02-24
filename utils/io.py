"""
I/O utilities: saving/loading checkpoints, configs and prediction CSVs.

Functions:
- save_checkpoint(path, state)
- load_checkpoint(path, map_location="cpu")
- save_config(path, config) (YAML or JSON)
- load_config(path)
- save_predictions(df, path)
"""
from pathlib import Path
import json
import yaml
import torch
import pandas as pd
from typing import Any, Dict

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    """
    Save a training checkpoint. 'state' is typically a dict containing:
      {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, ...}
    """
    p = Path(path)
    ensure_parent(p)
    torch.save(state, str(p))
    # no return; exception will propagate if fail

def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint with torch.load. Returns the saved dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return torch.load(str(p), map_location=map_location)

def save_config(path: str, config: Dict[str, Any], use_yaml: bool = True) -> None:
    """
    Save configuration dictionary to YAML (default) or JSON.
    """
    p = Path(path)
    ensure_parent(p)
    if use_yaml:
        with open(p, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    else:
        with open(p, "w") as f:
            json.dump(config, f, indent=2)

def load_config(path: str) -> Dict[str, Any]:
    """
    Load config from YAML or JSON depending on file extension.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if p.suffix in [".yaml", ".yml"]:
        with open(p, "r") as f:
            return yaml.safe_load(f)
    elif p.suffix == ".json":
        with open(p, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config format. Use .yaml/.yml or .json")

def save_predictions(df: pd.DataFrame, path: str) -> None:
    """
    Save predictions DataFrame to CSV. Overwrites if present.
    """
    p = Path(path)
    ensure_parent(p)
    df.to_csv(str(p), index=False)