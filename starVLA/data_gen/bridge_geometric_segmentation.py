"""
Simple gripper-driven segmentation for BRIDGE-LeRobot trajectories.

We assume each successful task follows a single grasp→place cycle:

0. Move to object: frames before the first gripper close event.
1. Move to goal: frames between the first close and the subsequent open.
2. Place object: frames from the open event to the end.

If any of these events is missing, we gracefully downgrade the quality tag
and fall back to the remaining segments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import numpy as np
import pyarrow.parquet as pq

DEFAULT_BRIDGE_ROOT = "/share/project/baishuanghao/data/bridge_orig_lerobot"


@dataclass
class Segment:
    segment_id: int
    segment_type: str
    start_t: int
    end_t: int
    geometry_debug: Dict[str, float]


PRIMITIVE_TYPES = [
    "move_to_object",
    "move_to_goal",
    "place_object",
]
SEGMENT_ID_MAP = {name: idx for idx, name in enumerate(PRIMITIVE_TYPES)}


def load_info(root: Path) -> Dict[str, Any]:
    info_path = root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_episode_indices(info: Dict[str, Any], split: str) -> Iterable[int]:
    split_spec = info.get("splits", {}).get(split)
    if split_spec is None:
        raise ValueError(f"Split '{split}' not found in info.json")
    if isinstance(split_spec, str) and ":" in split_spec:
        start_str, end_str = split_spec.split(":")
        return range(int(start_str), int(end_str))
    raise ValueError(f"Unsupported split spec: {split_spec}")


def load_episode(root: Path, episode_index: int, chunk_size: int) -> Dict[str, np.ndarray]:
    chunk_index = episode_index // chunk_size
    parquet_path = root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file missing: {parquet_path}")

    table = pq.read_table(parquet_path)
    data = {name: table[name].to_pylist() for name in table.column_names}
    state = np.stack(data["observation.state"])
    action = np.stack(data["action"])
    return {"state": state, "action": action}


def derive_gripper_signal(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    g_action = action[:, -1].astype(np.float32)
    g_state = state[:, -1].astype(np.float32)
    gripper = g_action if np.std(g_action) > 1e-4 else g_state
    if gripper.max() > gripper.min():
        gripper = (gripper - gripper.min()) / (gripper.ptp())
    else:
        gripper = np.zeros_like(gripper)
    return gripper


def binarize_gripper(gripper: np.ndarray, th_open: float, th_close: float) -> np.ndarray:
    g_bin = np.zeros_like(gripper, dtype=np.int32)
    g_bin[0] = 1 if gripper[0] >= th_close else 0
    for i in range(1, len(gripper)):
        val = gripper[i]
        if val >= th_close:
            g_bin[i] = 1
        elif val <= th_open:
            g_bin[i] = 0
        else:
            g_bin[i] = g_bin[i - 1]
    return g_bin


def detect_events(g_bin: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect gripper state transitions.
    Note: 1=Open, 0=Closed in Bridge data.
    
    Returns:
        opens: List of indices where gripper transitions Closed->Open (0->1) = place action
        closes: List of indices where gripper transitions Open->Closed (1->0) = grasp action
    """
    diff = np.diff(g_bin, prepend=g_bin[0])
    # 0->1: Closed to Open = place action
    opens = np.where(diff == 1)[0].tolist()
    # 1->0: Open to Closed = grasp action
    closes = np.where(diff == -1)[0].tolist()
    return opens, closes


def build_segments(T: int, grasp_idx: int, place_idx: int) -> List[Segment]:
    segments: List[Segment] = []

    if grasp_idx > 0:
        segments.append(
            Segment(
                segment_id=len(segments),
                segment_type="move_to_object",
                start_t=0,
                end_t=min(grasp_idx - 1, T - 1),
                geometry_debug={"duration": max(0, grasp_idx)},
            )
        )

    if place_idx > grasp_idx:
        segments.append(
            Segment(
                segment_id=len(segments),
                segment_type="move_to_goal",
                start_t=max(0, grasp_idx),
                end_t=min(place_idx - 1, T - 1),
                geometry_debug={"duration": max(0, place_idx - grasp_idx)},
            )
        )

    segments.append(
        Segment(
            segment_id=len(segments),
            segment_type="place_object",
            start_t=min(place_idx, T - 1),
            end_t=T - 1,
            geometry_debug={"duration": max(0, T - place_idx)},
            )
        )

    return segments


def build_dense_labels(T: int, segments: List[Segment]) -> Dict[str, List[int]]:
    subtask = [0] * T
    for seg in segments:
        seg_id = SEGMENT_ID_MAP.get(seg.segment_type, 0)
        for t in range(seg.start_t, seg.end_t + 1):
                if 0 <= t < T:
                    subtask[t] = seg_id
    return {"subtask_id": subtask, "cycle_id": [0] * T}


def process_episode(
    episode_index: int,
    root: Path,
    chunk_size: int,
    th_open: float,
    th_close: float,
) -> Dict[str, Any]:
    episode = load_episode(root, episode_index, chunk_size)
    state, action = episode["state"], episode["action"]
    T = state.shape[0]

    if T == 0:
        return {
            "dataset_root": str(root),
            "episode_index": episode_index,
            "num_steps": 0,
            "segmentation_method": "gripper_first_cycle_v1",
            "segmentation_quality": "empty_episode",
            "cycles": [],
            "dense_labels": {"subtask_id": [], "cycle_id": []},
        }

    gripper = derive_gripper_signal(state, action)
    g_bin = binarize_gripper(gripper, th_open, th_close)
    opens, closes = detect_events(g_bin)  # opens=0->1 (place), closes=1->0 (grasp)

    if not closes:
        dense = {"subtask_id": [0] * T, "cycle_id": [0] * T}
        return {
            "dataset_root": str(root),
            "episode_index": episode_index,
            "num_steps": T,
            "segmentation_method": "gripper_first_cycle_v1",
            "segmentation_quality": "no_grasp",
            "cycles": [],
            "dense_labels": dense,
        }

    # First close = grasp action (1->0 transition)
    grasp_idx = closes[0]
    # Find first open after grasp = place action (0->1 transition)
    place_candidates = [o for o in opens if o > grasp_idx]
    if place_candidates:
        place_idx = place_candidates[0]
        quality = "high"
    else:
        place_idx = T - 1
        quality = "missing_place"

    segments = build_segments(T, grasp_idx, place_idx)
    dense = build_dense_labels(T, segments)
    cycles = [
        {
            "cycle_id": 0,
            "segments": [asdict(seg) for seg in segments],
        }
    ]

    return {
        "dataset_root": str(root),
        "episode_index": episode_index,
        "num_steps": T,
        "segmentation_method": "gripper_first_cycle_v1",
        "segmentation_quality": quality,
        "cycles": cycles,
        "dense_labels": dense,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple gripper-based segmentation")
    parser.add_argument("--bridge_root", type=str, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="intermediate/geo_segments.jsonl")
    parser.add_argument("--th_open", type=float, default=0.3)
    parser.add_argument("--th_close", type=float, default=0.7)
    parser.add_argument("--max_episodes", type=int, default=-1)
    args = parser.parse_args()

    root = Path(args.bridge_root)
    info = load_info(root)
    chunk_size = int(info["chunks_size"])
    indices = list(iter_episode_indices(info, args.split))
    if args.max_episodes > 0:
        indices = indices[: args.max_episodes]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, ep in enumerate(indices):
            try:
                result = process_episode(
                    episode_index=ep,
                    root=root,
                    chunk_size=chunk_size,
                    th_open=args.th_open,
                    th_close=args.th_close,
                )
            except Exception as e:
                result = {
                    "dataset_root": str(root),
                    "episode_index": ep,
                    "num_steps": 0,
                    "segmentation_method": "gripper_first_cycle_v1",
                    "segmentation_quality": "error",
                    "error": str(e),
                    "cycles": [],
                    "dense_labels": {"subtask_id": [], "cycle_id": []},
                }
            f.write(json.dumps(result) + "\n")
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} episodes...")

    print(f"Segmentation saved to {output_path}")


if __name__ == "__main__":
    main()

