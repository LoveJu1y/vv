#!/usr/bin/env python3
"""
Microbenchmark: Bridge (LeRobot) explicit forward latency.

Measures model compute only:
  1) Explicit thinking generation (generate_thinking_explicit)
  2) Full forward on generated ids to get hidden states
  3) Action head predict_action

Excludes dataloader/video decode/tokenization/image preprocessing by caching
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
        default="/share/project/lvjing/starVLA/results/BridgeFinal_Action/bridge_explicit_cot_stage1_final/config.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/share/project/lvjing/starVLA/results/BridgeFinal_Action/bridge_explicit_cot_stage1_final/checkpoints/steps_40000_pytorch_model.pt",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("bf16", "fp16"),
        default="bf16",
        help="Autocast dtype for VLM compute in the benchmark region.",
    )
    parser.add_argument("--max_thinking_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--print_thinking", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config_yaml)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended to run on CUDA.")

    # Build model
    model = build_framework(cfg).to(device)
    model.eval()

    # Load checkpoint
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

    # Fetch ONE sample (not timed).
    dataset = get_vla_dataset(
        cfg.datasets.vla_data,
        mode="train",
        delete_pause_frame=cfg.datasets.vla_data.delete_pause_frame,
    )
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
        img_next_mask_prompt = (
            (qwen_inputs["input_ids"] == img_next_token_id) if img_next_token_id is not None else None
        )

    vlm_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    times_gen_ms: list[float] = []
    times_forward_ms: list[float] = []
    times_action_ms: list[float] = []
    times_total_ms: list[float] = []
    gen_new_tokens: list[int] = []

    def _run_once() -> None:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=vlm_dtype):
            gen_result = model.qwen_vl_interface.generate_thinking_explicit(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                max_thinking_len=args.max_thinking_len,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        thinking_texts = gen_result.get("thinking_text", [])
        if args.print_thinking and thinking_texts:
            print(f"[explicit thinking] sample0: {thinking_texts[0]}")

        full_ids = gen_result["generated_ids"]
        full_mask = torch.ones_like(full_ids)

        with torch.autocast("cuda", dtype=vlm_dtype):
            qwenvl_outputs = model.qwen_vl_interface(
                input_ids=full_ids,
                attention_mask=full_mask,
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        last_hidden = qwenvl_outputs.hidden_states[-1]

        # Match current QwenGR00T.predict_action behavior: pass prompt-length img_next_mask.
        with torch.autocast("cuda", dtype=torch.float32):
            _ = model.action_model.predict_action(
                vl_embs=last_hidden,
                state=None,
                reasoning_mask=None,
                img_next_mask=img_next_mask_prompt,
            )
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        times_gen_ms.append((t1 - t0) * 1000.0)
        times_forward_ms.append((t2 - t1) * 1000.0)
        times_action_ms.append((t3 - t2) * 1000.0)
        times_total_ms.append((t3 - t0) * 1000.0)
        gen_new_tokens.append(int(full_ids.shape[1] - qwen_inputs["input_ids"].shape[1]))

    print("=== Bridge explicit forward benchmark ===")
    print(f"config_yaml: {args.config_yaml}")
    print(f"ckpt:        {args.ckpt}")
    print(f"vlm dtype:   {args.dtype} (autocast)")
    print(f"action dtype: float32 (autocast)")
    print(f"warmup:      {args.warmup}")
    print(f"steps:       {args.steps}")
    print(f"max_thinking_len: {args.max_thinking_len} temperature={args.temperature} top_p={args.top_p}")
    print(f"img_next_token_id: {getattr(model.qwen_vl_interface, 'img_next_token_id', None)}")
    print(f"prompt input_ids shape: {tuple(qwen_inputs['input_ids'].shape)}")
    if qwen_inputs.get("pixel_values") is not None:
        print(f"pixel_values shape:     {tuple(qwen_inputs['pixel_values'].shape)}")

    # Warmup (not recorded)
    with torch.inference_mode():
        for _ in range(args.warmup):
            _run_once()

    times_gen_ms.clear()
    times_forward_ms.clear()
    times_action_ms.clear()
    times_total_ms.clear()
    gen_new_tokens.clear()

    with torch.inference_mode():
        for _ in range(args.steps):
            _run_once()

    print("--- Results (ms) ---")
    print("Thinking generate: ", _as_percentiles(times_gen_ms))
    print("VLM full forward:  ", _as_percentiles(times_forward_ms))
    print("Action predict:    ", _as_percentiles(times_action_ms))
    print("Total:             ", _as_percentiles(times_total_ms))
    if gen_new_tokens:
        toks = np.asarray(gen_new_tokens, dtype=np.int64)
        print(
            f"Generated new tokens: mean={float(toks.mean()):.1f} "
            f"p50={int(np.percentile(toks, 50))} p90={int(np.percentile(toks, 90))} "
            f"min={int(toks.min())} max={int(toks.max())}"
        )


if __name__ == "__main__":
    main()

