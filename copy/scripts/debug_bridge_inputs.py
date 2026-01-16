#!/usr/bin/env python3
"""
Quick utility to iterate Bridge-LeRobot samples and run Qwen3-VL input building
without launching full training. This triggers the same alignment / debug
prints as train_ecot so issues such as image_pad mismatches can be reproduced
quickly.
"""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.train_ecot import sync_bridge_reasoning_to_framework


def parse_args():
    parser = argparse.ArgumentParser(description="Debug Bridge dataset -> Qwen3 inputs.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/bridge_lerobot_stage2.yaml",
        help="Path to training config with datasets.* definitions.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Number of batches to inspect before exiting (None = full dataset).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Optional override for datasets.vla_data.per_device_batch_size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for loading QwenGR00T.",
    )
    return parser.parse_args()


def ensure_output_dir(cfg):
    if not hasattr(cfg, "run_root_dir"):
        cfg.run_root_dir = "debug_outputs"
    if not hasattr(cfg, "run_id"):
        cfg.run_id = "debug_bridge_inputs"
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_yaml)

    if args.batch_size is not None:
        cfg.datasets.vla_data.per_device_batch_size = args.batch_size

    ensure_output_dir(cfg)
    sync_bridge_reasoning_to_framework(cfg)

    print(f"[Debug] Loading model framework on {args.device} ...")
    model = build_framework(cfg).to(args.device)
    model.eval()
    qwen_vl = model.qwen_vl_interface

    print("[Debug] Building dataloader ...")
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    target_batches = args.max_batches
    if target_batches is None:
        print("[Debug] Scanning entire dataset (no batch limit) ...")
    else:
        print(f"[Debug] Scanning up to {target_batches} batches ...")
    for batch_idx, batch in enumerate(dataloader):
        batch_images = [sample["image"] for sample in batch]
        instructions = [sample["lang"] for sample in batch]
        try:
            _ = qwen_vl.build_qwenvl_inputs(images=batch_images, instructions=instructions)
            print(f"[Debug] Batch {batch_idx} processed successfully.")
        except Exception as exc:
            print(f"[Debug] Batch {batch_idx} raised exception: {exc}")
            raise

        if target_batches is not None and (batch_idx + 1) >= target_batches:
            break

    print("[Debug] Completed scan without triggering errors.")


if __name__ == "__main__":
    main()
