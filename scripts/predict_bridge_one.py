#!/usr/bin/env python3
"""
Load Bridge dataset, take the first sample, and run model.predict_action once.
Defaults mirror scripts/benchmark_bridge_implicit_forward.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework
from starVLA.model.modules.action_model.flow_matching_head import cross_attention_dit as dit_debug


def _load_state_dict(path: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("bf16", "fp16"),
        default="bf16",
        help="Autocast dtype for the predict region.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config_yaml)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(cfg).to(device)
    model.eval()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = _load_state_dict(args.ckpt)
    state_dict = _strip_prefix(state_dict, "module.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (showing first 10): {missing[:10]}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (showing first 10): {unexpected[:10]}")

    dataset = get_vla_dataset(
        cfg.datasets.vla_data,
        mode="train",
        delete_pause_frame=cfg.datasets.vla_data.delete_pause_frame,
    )
    sample = dataset[2]

    batch_images = [sample["image"]]
    instructions = [sample["lang"]]
    state = None
    if "state" in sample:
        state = np.asarray(sample["state"])

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        result = model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            state=state,
        )

    if getattr(dit_debug, "DEBUG_THINKING_ATTN", False):
        try:
            cache = model.action_model.model.get_and_clear_thinking_attn_cache()
            if cache:
                out_dir = Path("/share/project/lvjing/starVLA/results/ANALY")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "thinking_attn_cache.pt"
                torch.save(cache, out_path)
                print(f"[thinking_attn] saved: {out_path}")
        except Exception as exc:
            print(f"[thinking_attn] failed to save cache: {exc}")

    actions = result.get("normalized_actions", None)
    if actions is None:
        raise RuntimeError("predict_action did not return normalized_actions.")
    print(f"predict_action output shape: {actions.shape}")


if __name__ == "__main__":
    main()
