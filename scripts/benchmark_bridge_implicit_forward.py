#!/usr/bin/env python3
"""
Microbenchmark: Bridge (LeRobot) implicit forward latency.

Measures model compute only (VLM forward_latent + action head predict_action),
excluding dataloader/video decode/tokenization/image preprocessing by caching
the prepared Qwen inputs once before timing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework


def _as_percentiles(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _load_state_dict(path: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        # Common patterns:
        # - raw state_dict
        # - {"state_dict": ...}
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        return state
    raise ValueError(f"Unsupported checkpoint format: {type(state)}")


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not prefix:
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/bridge_lerobot_stage2.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/share/project/lvjing/starVLA/results/BridgeFinal_Action/SDPA5_bridge_lerobot_DITB_LR1E-4_LR1E-5_BTS16_60K_FINAL_NO_IMGLOSS__1.3lr/checkpoints/steps_15000_pytorch_model.pt",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("bf16", "fp16"),
        default="bf16",
        help="Autocast dtype for the benchmark region.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config_yaml)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(cfg).to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = _load_state_dict(args.ckpt)
    # Handle common DDP prefixes.
    state_dict = _strip_prefix(state_dict, "module.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (showing first 10): {missing[:10]}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (showing first 10): {unexpected[:10]}")

    # Build dataset and fetch ONE sample (not timed).
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode="train", delete_pause_frame=cfg.datasets.vla_data.delete_pause_frame)
    sample = dataset[0]
    examples = [sample]

    # Precompute Qwen inputs once (not timed): tokenization + image preprocessing are excluded.
    with torch.inference_mode():
        qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=[examples[0]["image"]],
            instructions=[examples[0]["lang"]],
            action_tokens=[examples[0].get("action_tokens", "")],
        )
        img_next_token_id = getattr(model.qwen_vl_interface, "img_next_token_id", None)
        img_next_mask = (
            (qwen_inputs["input_ids"] == img_next_token_id) if img_next_token_id is not None else None
        )
        reasoning_mask = model._extract_reasoning_mask(qwen_inputs)

    # Benchmark region: VLM forward_latent + action head predict_action
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    times_vlm_ms: list[float] = []
    times_action_ms: list[float] = []
    times_total_ms: list[float] = []

    def _run_once() -> None:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            vlm_out = model.qwen_vl_interface.forward_latent(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                labels=None,
                position_ids=qwen_inputs.get("position_ids"),
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        last_hidden = vlm_out["hidden_states"]
        with torch.autocast("cuda", dtype=dtype):
            _ = model.action_model.predict_action(
                vl_embs=last_hidden,
                state=None,
                reasoning_mask=reasoning_mask,
                img_next_mask=img_next_mask,
            )
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        vlm_ms = (t1 - t0) * 1000.0
        act_ms = (t2 - t1) * 1000.0
        tot_ms = (t2 - t0) * 1000.0
        times_vlm_ms.append(vlm_ms)
        times_action_ms.append(act_ms)
        times_total_ms.append(tot_ms)

    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended to run on CUDA.")

    print("=== Bridge implicit forward benchmark ===")
    print(f"config_yaml: {args.config_yaml}")
    print(f"ckpt:        {args.ckpt}")
    print(f"dtype:       {args.dtype}")
    print(f"warmup:      {args.warmup}")
    print(f"steps:       {args.steps}")
    print(f"img_next_token_id: {getattr(model.qwen_vl_interface, 'img_next_token_id', None)}")
    print(f"input_ids shape:   {tuple(qwen_inputs['input_ids'].shape)}")
    if qwen_inputs.get("pixel_values") is not None:
        print(f"pixel_values shape:{tuple(qwen_inputs['pixel_values'].shape)}")

    # Warmup (not recorded)
    with torch.inference_mode():
        for _ in range(args.warmup):
            _run_once()

    # Clear warmup measurements
    times_vlm_ms.clear()
    times_action_ms.clear()
    times_total_ms.clear()

    # Measure
    with torch.inference_mode():
        for _ in range(args.steps):
            _run_once()

    print("--- Results (ms) ---")
    print("VLM forward_latent:", _as_percentiles(times_vlm_ms))
    print("Action predict:    ", _as_percentiles(times_action_ms))
    print("Total:             ", _as_percentiles(times_total_ms))


if __name__ == "__main__":
    main()

