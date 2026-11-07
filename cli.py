"""Consistent command-line options for operation and point recognition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from agri.model import VARIANTS
from model import POINT_VARIANTS
from training import evaluate_checkpoint, predict_files, train_experiment


def make_parser(task="pattern"):
    parser = argparse.ArgumentParser(description="Field trajectory recognition: " + task)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "eval", "predict"):
        action = commands.add_parser(command)
        action.add_argument("--output", type=Path, required=True, help="New or empty run directory")
        action.add_argument("--device", default="auto")
        action.add_argument("--batch-size", type=int, default=1)
        action.add_argument("--threads", type=int, default=2)
        if command == "train":
            action.add_argument("--data-root", type=Path, required=True)
            action.add_argument("--seed", type=int, default=42)
            action.add_argument("--epochs", type=int, default=80 if task == "pattern" else 100)
            action.add_argument("--learning-rate", type=float, default=0.001 if task == "pattern" else 0.0001)
            action.add_argument("--weight-decay", type=float, default=0 if task == "pattern" else 0.001)
            action.add_argument("--lr-step", type=int, default=5 if task == "pattern" else 10)
            action.add_argument("--lr-gamma", type=float, default=0.1 if task == "pattern" else 0.9)
            action.add_argument("--patience", type=int, default=10)
            action.add_argument("--min-delta", type=float, default=0)
            action.add_argument("--dropout", type=float, default=0.1)
            action.add_argument("--lstm-layers", type=int, default=1)
            action.add_argument("--validation-fraction", type=float, default=0.2)
            action.add_argument("--test-fraction", type=float, default=0.1)
            action.add_argument("--split-policy", choices=("stratified", "provided"), default="stratified")
            action.add_argument(
                "--split-manifest", type=Path, help="Use the exact file partitions from an existing run"
            )
            action.add_argument("--allow-missing-classes", action="store_true")
            action.add_argument("--limit-per-class", type=int)
            action.add_argument("--interval-seconds", type=float, default=4)
            action.add_argument("--stop-speed-kmh", type=float, default=0.1)
            action.add_argument("--max-speed-kmh", type=float, default=60)
            action.add_argument("--max-gap-seconds", type=float, default=12)
            action.add_argument("--label-column")
            if task == "pattern":
                action.add_argument("--variant", choices=VARIANTS, default="full")
                action.add_argument("--graph-size", type=int, default=160)
                action.add_argument("--local-nodes", type=int, default=8)
                action.add_argument("--hidden-dim", type=int, default=64)
            else:
                action.add_argument("--variant", choices=POINT_VARIANTS, default="improved")
                action.add_argument("--d-model", type=int, default=128)
                action.add_argument("--nhead", type=int, default=4)
                action.add_argument("--num-layers", type=int, default=3)
                action.add_argument(
                    "--local-window",
                    type=int,
                    default=6,
                    help="Symmetric attention radius in sampled time steps",
                )
                action.add_argument("--time-decay", type=float, default=0.1)
        else:
            action.add_argument("--checkpoint", type=Path, required=True)
            if command == "eval":
                action.add_argument("--data-root", type=Path, required=True)
                action.add_argument("--manifest", type=Path, required=True)
                action.add_argument("--split", choices=("train", "val", "test"), default="test")
            else:
                action.add_argument("--input", type=Path, required=True)
    return parser


def main(task="pattern", argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = make_parser(task)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) < 1:
        parser.error("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    handler = {"train": train_experiment, "eval": evaluate_checkpoint, "predict": predict_files}[args.command]
    try:
        handler(args, task)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0
