from enum import Enum
from typing import Dict


class CotMode(str, Enum):
    NONE = "none"  # 无 CoT 数据，无思维生成
    VLM_SEEN_NO_OUT = "vlm_seen_no_out"  # VLM 见过 CoT，推理不输出
    EXPLICIT = "explicit"  # 显式自回归生成 + tokens 进 cross-attn
    IMPLICIT = "implicit"  # 隐式推理：hidden → summarizer/FiLM


def parse_cot_mode(cfg) -> CotMode:
    """
    从配置解析 cot_mode，缺省为 implicit（保持兼容）
    """
    raw = getattr(getattr(cfg, "framework", {}), "cot_mode", "implicit")
    return CotMode(raw.lower())


def derive_flags_from_mode(mode: CotMode) -> Dict[str, object]:
    """
    根据模式派生训练/推理需要的布尔开关和阶段设置
    """
    return {
        "enable_latent_reasoning": mode == CotMode.IMPLICIT,
        "emit_thinking_tokens": False if mode == CotMode.EXPLICIT else (mode == CotMode.IMPLICIT),
        "use_iterative_forward": mode == CotMode.IMPLICIT,
        "generate_thinking": mode in (CotMode.EXPLICIT, CotMode.IMPLICIT),
        "reasoning_stage": {
            CotMode.NONE: 0,
            CotMode.VLM_SEEN_NO_OUT: 1,
            CotMode.EXPLICIT: 1,
            CotMode.IMPLICIT: 4,
        }[mode],
    }
