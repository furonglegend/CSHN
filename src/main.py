#!/usr/bin/env python3
"""
Command-line entry point for training / evaluating / inferring with SphUnc toy implementation.

Usage examples:
  python src/main.py --mode train --config configs/default.yaml
  python src/main.py --mode eval  --config configs/default.yaml
  python src/main.py --mode infer --config configs/default.yaml --input examples/sample.json
"""
import argparse
import os
import sys
from pathlib import Path
import yaml

# Make project root importable
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from train import run_training
from eval import run_evaluation
from infer import run_inference

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval", "infer"], required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--input", type=str, default=None, help="input JSON or CSV for inference")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found. Using defaults.")
    if args.mode == "train":
        run_training(config)
    elif args.mode == "eval":
        run_evaluation(config)
    elif args.mode == "infer":
        if args.input is None:
            raise ValueError("Please provide --input for inference.")
        run_inference(config, args.input)

if __name__ == "__main__":
    main()