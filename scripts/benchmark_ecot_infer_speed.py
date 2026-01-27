#!/usr/bin/env python3
import argparse
import os
import time
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.utils import import_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ECOT OpenVLA inference latency."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/share/project/lvjing/models/hub/models--Embodied-CoT--ecot-openvla-7b-bridge",
        help="Local path to the ECOT model directory.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/share/project/baishuanghao/data/bridge_orig_lerobot/videos_decoded/chunk_0/episode_000002/00000.png",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="put the red object into the pot",
        help="Instruction for the robot.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of measured runs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (not timed).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g., cuda or cpu).",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str) -> str:
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path

    snapshots_dir = os.path.join(model_path, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_path

    snapshot_names = sorted(os.listdir(snapshots_dir))
    for name in reversed(snapshot_names):
        candidate = os.path.join(snapshots_dir, name)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return model_path


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    resolved_model_path = resolve_model_path(args.model_path)

    # Prevent transformers from importing flash-attn (can fail on older glibc).
    original_is_package_available = import_utils._is_package_available

    def _patched_is_package_available(pkg_name: str, return_version: bool = False):
        if pkg_name == "flash_attn":
            return (False, "N/A") if return_version else False
        return original_is_package_available(pkg_name, return_version=return_version)

    import_utils._is_package_available = _patched_is_package_available

    # Prefer SDPA to avoid flash-attn binary dependency issues.
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    processor = AutoProcessor.from_pretrained(
        resolved_model_path, trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        resolved_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    vla.eval()

    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: What action should the robot take to {args.instruction.lower()}? "
        "ASSISTANT: TASK:"
    )
    image = Image.open(args.image_path).convert("RGB")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup (includes preprocessing to mimic end-to-end path)
    with torch.inference_mode():
        for _ in range(args.warmup):
            inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
            _ = vla.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                max_new_tokens=args.max_new_tokens,
            )
            sync()

    # Timed runs
    latencies = []
    preprocess_latencies = []
    with torch.inference_mode():
        for _ in range(args.runs):
            sync()
            start = time.perf_counter()
            inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
            preprocess_latencies.append(time.perf_counter() - start)

            sync()
            start = time.perf_counter()
            _ = vla.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                max_new_tokens=args.max_new_tokens,
            )
            sync()
            latencies.append(time.perf_counter() - start)

    def percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        if pct <= 0:
            return min(values)
        if pct >= 100:
            return max(values)
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (pct / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == c:
            return sorted_vals[f]
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

    def summarize(values: List[float]):
        return {
            "avg": sum(values) / len(values) if values else 0.0,
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    preprocess_stats = summarize(preprocess_latencies)
    infer_stats = summarize(latencies)
    end_to_end_latencies = [
        p + i for p, i in zip(preprocess_latencies, latencies)
    ]
    end_to_end_stats = summarize(end_to_end_latencies)

    print(f"Runs: {len(latencies)}")
    print(
        "Preprocess latency (s) "
        f"avg={preprocess_stats['avg']:.4f} "
        f"p50={preprocess_stats['p50']:.4f} "
        f"p90={preprocess_stats['p90']:.4f} "
        f"p95={preprocess_stats['p95']:.4f} "
        f"p99={preprocess_stats['p99']:.4f} "
        f"min={preprocess_stats['min']:.4f} "
        f"max={preprocess_stats['max']:.4f}"
    )
    print(
        "Infer latency (s) "
        f"avg={infer_stats['avg']:.4f} "
        f"p50={infer_stats['p50']:.4f} "
        f"p90={infer_stats['p90']:.4f} "
        f"p95={infer_stats['p95']:.4f} "
        f"p99={infer_stats['p99']:.4f} "
        f"min={infer_stats['min']:.4f} "
        f"max={infer_stats['max']:.4f}"
    )
    print(
        "End-to-end latency (s) "
        f"avg={end_to_end_stats['avg']:.4f} "
        f"p50={end_to_end_stats['p50']:.4f} "
        f"p90={end_to_end_stats['p90']:.4f} "
        f"p95={end_to_end_stats['p95']:.4f} "
        f"p99={end_to_end_stats['p99']:.4f} "
        f"min={end_to_end_stats['min']:.4f} "
        f"max={end_to_end_stats['max']:.4f}"
    )


if __name__ == "__main__":
    main()
