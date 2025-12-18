"""
Apply SAM3-based bounding box detection to segments obtained from geometric
segmentation. This script expects a JSONL file (e.g., intermediate/geo_segments.jsonl)
and augments each segment with a `grounding` field containing the detected bbox.

SAM3 integration is implementation-dependent. By default, we expect a python package
named `sam3` that exposes a detector with a `.detect(image, text_prompt, topk)`
API. If such package is not available, the script will raise an ImportError with
instructions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("opencv-python is required for frame extraction") from exc


DEFAULT_BRIDGE_ROOT = "/share/project/baishuanghao/data/bridge_orig_lerobot"


def load_info(root: Path) -> Dict[str, Any]:
    info_path = root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(tasks_path: Path) -> Dict[int, str]:
    tasks = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            tasks[int(item["task_index"])] = item.get("task", "")
    return tasks


def load_episode_tasks(episodes_path: Path) -> Dict[int, int]:
    mapping = {}
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            mapping[int(item["episode_index"])] = int(item["task_index"])
    return mapping


def load_geo_segments(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class Sam3Detector:
    """Thin wrapper around an external SAM3 bounding-box detector."""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        try:
            from sam3 import Sam3Detector as _Sam3Detector  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on user env
            raise ImportError(
                "sam3 package is not installed. Please install the official SAM3 "
                "implementation and ensure it exposes Sam3Detector."
            ) from exc

        self.detector = _Sam3Detector(checkpoint=checkpoint, device=device)

    def predict(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        topk: int = 1,
        score_threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        detections = self.detector.detect(
            image=image_bgr,
            text_prompt=prompt,
            topk=topk,
        )
        results = []
        for det in detections:
            score = float(det.get("score", 0.0))
            if score < score_threshold:
                continue
            bbox = det.get("bbox")
            if bbox is None:
                continue
            results.append(
                {
                    "bbox": bbox,
                    "score": score,
                }
            )
            if len(results) >= topk:
                break
        return results


def select_keyframe(segment: Dict[str, Any], strategy: str = "end") -> int:
    start_t = int(segment["start_t"])
    end_t = int(segment["end_t"])
    if strategy == "middle":
        return (start_t + end_t) // 2
    if strategy == "start":
        return start_t
    return end_t


def load_frame_image(
    root: Path,
    info: Dict[str, Any],
    camera_key: str,
    episode_index: int,
    frame_idx: int,
) -> np.ndarray:
    chunk_size = int(info["chunks_size"])
    chunk_index = episode_index // chunk_size
    video_path_pattern = info["video_path"]
    video_rel = video_path_pattern.format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key=camera_key,
    )
    video_path = root / video_rel
    if not video_path.exists():
        raise FileNotFoundError(f"Video file missing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    if not success or frame is None:
        raise RuntimeError(
            f"Failed to read frame {frame_idx} from {video_path}"
        )
    return frame


def choose_prompt(
    segment: Dict[str, Any],
    default_prompt: Optional[str],
    instruction: str,
) -> Optional[str]:
    if "target_object_ref" in segment and segment["target_object_ref"]:
        return segment["target_object_ref"]
    if default_prompt:
        return default_prompt
    return instruction or None


def build_dense_active_bbox(num_steps: int, cycles: List[Dict[str, Any]]) -> List[Optional[List[float]]]:
    dense: List[Optional[List[float]]] = [None] * num_steps
    for cycle in cycles:
        for seg in cycle.get("segments", []):
            grounding = seg.get("grounding")
            if not grounding or "bbox_2d" not in grounding:
                continue
            bbox = grounding["bbox_2d"]
            for t in range(seg["start_t"], seg["end_t"] + 1):
                if 0 <= t < num_steps:
                    dense[t] = bbox
    return dense


def annotate_episode(
    record: Dict[str, Any],
    detector: Sam3Detector,
    root: Path,
    info: Dict[str, Any],
    instruction: str,
    camera_key: str,
    default_prompt: Optional[str],
    keyframe_strategy: str,
) -> Dict[str, Any]:
    episode_index = int(record["episode_index"])
    for cycle in record.get("cycles", []):
        for segment in cycle.get("segments", []):
            prompt = choose_prompt(segment, default_prompt, instruction)
            if not prompt:
                continue
            frame_idx = select_keyframe(segment, strategy=keyframe_strategy)
            image = load_frame_image(root, info, camera_key, episode_index, frame_idx)
            preds = detector.predict(image, prompt)
            if not preds:
                continue
            bbox = preds[0]["bbox"]
            score = preds[0]["score"]
            segment["grounding"] = {
                "bbox_2d": bbox,
                "confidence": score,
                "frame_idx": frame_idx,
                "prompt": prompt,
            }

    dense_bbox = build_dense_active_bbox(record["num_steps"], record.get("cycles", []))
    record.setdefault("dense_labels", {})
    record["dense_labels"]["active_bbox"] = dense_bbox
    record["grounding_model"] = "sam3"
    return record


def main():
    parser = argparse.ArgumentParser(description="Apply SAM3 bbox detection to segments")
    parser.add_argument("--bridge_root", type=str, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--geo_segments", type=str, default="intermediate/geo_segments.jsonl")
    parser.add_argument("--output", type=str, default="annotations/segmentation_with_sam3.jsonl")
    parser.add_argument("--camera_key", type=str, default="observation.images.image_0")
    parser.add_argument("--keyframe_strategy", type=str, default="end", choices=["start", "middle", "end"])
    parser.add_argument("--default_prompt", type=str, default=None)
    parser.add_argument("--sam3_checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks_path", type=str, default="meta/tasks.jsonl")
    parser.add_argument("--episodes_path", type=str, default="meta/episodes.jsonl")
    args = parser.parse_args()

    root = Path(args.bridge_root)
    info = load_info(root)
    tasks = load_tasks(root / args.tasks_path)
    episode_task = load_episode_tasks(root / args.episodes_path)
    detector = Sam3Detector(checkpoint=args.sam3_checkpoint, device=args.device)

    geo_path = Path(args.geo_segments)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for record in load_geo_segments(geo_path):
            episode_index = int(record["episode_index"])
            task_idx = episode_task.get(episode_index)
            instruction = tasks.get(task_idx, "") if task_idx is not None else ""
            updated = annotate_episode(
                record=record,
                detector=detector,
                root=root,
                info=info,
                instruction=instruction,
                camera_key=args.camera_key,
                default_prompt=args.default_prompt,
                keyframe_strategy=args.keyframe_strategy,
            )
            fout.write(json.dumps(updated) + "\n")

    print(f"SAM3 annotations saved to {output_path}")


if __name__ == "__main__":
    main()

