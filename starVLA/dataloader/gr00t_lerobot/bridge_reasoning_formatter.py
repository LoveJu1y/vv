from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
import numpy as np


@dataclass
class BridgeReasoningFormatter:
    """
    Format per-step CoT / bbox metadata into training text.

    Stage 0: human-readable CoT text.
    Stage 1: structured tags ([SUBTASK] ... [/SUBTASK]).
    Stage 2+: thinking tokens (<|start_of_thinking|> <|thinking|> ... <|end_of_thinking|>).
    """

    stage: int = 0
    include_bbox: bool = True
    include_action_tokens: bool = True
    thinking_token: str = "<|thinking|>"
    start_token: str = "<|start_of_thinking|>"
    end_token: str = "<|end_of_thinking|>"
    tag2think_count: Optional[dict] = None
    component_order: Optional[Sequence[str]] = None

    def __post_init__(self):
        default_order = ("BBOX", "SUBTASK", "REASON")
        if self.component_order:
            normalized = []
            for tag in self.component_order:
                name = str(tag).strip().upper()
                if name in {"BBOX", "SUBTASK", "REASON"} and name not in normalized:
                    normalized.append(name)
            self.component_order = tuple(normalized) if normalized else default_order
        else:
            self.component_order = default_order

    def _bbox_to_str(self, bbox: np.ndarray, confidence: float) -> str:
        coords = " ".join(f"{float(x):.4f}" for x in bbox.tolist())
        # if confidence > 0:
        #     return f"{coords} (conf={confidence:.3f})"
        return coords

    def format(self, instruction: str, sample: dict) -> str:
        instruction = (instruction or "").strip()
        subtask = sample.get("cot_subtask", "") or ""
        reasoning = sample.get("cot_reasoning", "") or ""
        bbox = sample.get("bbox")
        bbox_valid = bool(sample.get("bbox_valid", False))
        bbox_conf = float(sample.get("bbox_confidence", 0.0))
        action_tokens = sample.get("action_tokens", "") or ""

        # Stage 0: plain text CoT
        if self.stage <= 1:
            return self._format_stage1(
                instruction=instruction,
                subtask=subtask,
                reasoning=reasoning,
                bbox=bbox if bbox_valid else None,
                bbox_conf=bbox_conf,
                action_tokens=action_tokens,
            )
        # Stage >=2: progressively convert components into thinking tokens
        else:
            latent_tags = self._determine_latent_tags_for_stage(self.stage)
            return self._format_latent(
                instruction=instruction,
                subtask=subtask,
                reasoning=reasoning,
                bbox=bbox if bbox_valid else None,
                bbox_conf=bbox_conf,
                latent_tags=latent_tags,
                action_tokens=action_tokens,
            )

    def _format_stage1(
        self,
        instruction: str,
        subtask: str,
        reasoning: str,
        bbox: Optional[np.ndarray],
        bbox_conf: float,
        action_tokens: str,
    ) -> str:
        parts = []
        for tag, value in self._iter_components(bbox, subtask, reasoning):
            parts.append(self._format_explicit_tag(tag, value, bbox_conf))
        if self.include_action_tokens and action_tokens:
            parts.append(f"Action: {action_tokens}")

        body = " ".join(parts) if parts else ""
        if instruction:
            delimiter = " @ "
            body_text = body if body else ""
            return f"{instruction}.{delimiter}{body_text}"
        return body

    def _format_latent(
        self,
        instruction: str,
        subtask: str,
        reasoning: str,
        bbox: Optional[np.ndarray],
        bbox_conf: float,
        latent_tags: set,
        action_tokens: str,
    ) -> str:
        segments = list(self._iter_components(bbox, subtask, reasoning))
        if self.include_action_tokens and action_tokens:
            segments.append(("ACTION", action_tokens))

        tag_counts = self.tag2think_count or {}
        thinking_body = ""
        explicit_parts = []

        for tag, value in segments:
            if tag in latent_tags:
                count = max(1, int(tag_counts.get(tag, 1)))
                thinking_body += self.thinking_token * count
            else:
                explicit_parts.append(self._format_explicit_tag(tag, value, bbox_conf))

        if thinking_body:
            thinking_span = f"{self.start_token}{thinking_body}{self.end_token}"
            if instruction:
                text = f"{instruction}. @ {thinking_span}"
            else:
                text = thinking_span
        else:
            text = instruction

        if explicit_parts:
            text = f"{text} " + " ".join(explicit_parts) if text else " ".join(explicit_parts)
        return text.strip()

    def _make_latent_span(self, tag: str) -> str:
        tag_counts = self.tag2think_count or {}
        count = int(tag_counts.get(tag, 1))
        count = max(count, 1)
        body = self.thinking_token * count
        return f"{self.start_token}{body}{self.end_token}"

    def _determine_latent_tags_for_stage(self, stage: int) -> set:
        latent_tags = set()
        if stage >= 2:
            latent_tags.add("BBOX")
        if stage >= 3:
            latent_tags.add("SUBTASK")
        if stage >= 4:
            latent_tags.add("REASON")
        return latent_tags

    def _format_explicit_tag(self, tag: str, value, bbox_conf: float) -> str:
        if tag == "SUBTASK":
            return f"Subtask: {value}."
        if tag == "REASON":
            return f"Reasoning: {value}"
        if tag == "BBOX":
            return f"BBox: [{self._bbox_to_str(value, bbox_conf)}]."
        if tag == "ACTION":
            return f"Action: {value}"
        return str(value)

    def _iter_components(
        self,
        bbox: Optional[np.ndarray],
        subtask: str,
        reasoning: str,
    ):
        order: Tuple[str, ...] = self.component_order  # type: ignore[assignment]
        subtask = subtask or ""
        reasoning = reasoning or ""
        for tag in order:
            if tag == "BBOX":
                if self.include_bbox and bbox is not None:
                    yield ("BBOX", bbox)
            elif tag == "SUBTASK":
                if subtask:
                    yield ("SUBTASK", subtask)
            elif tag == "REASON":
                if reasoning:
                    yield ("REASON", reasoning)
