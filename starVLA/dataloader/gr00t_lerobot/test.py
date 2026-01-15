# run_fmt_demo.py
from starVLA.dataloader.gr00t_lerobot.bridge_reasoning_formatter import BridgeReasoningFormatter
import numpy as np

def dump(stage):
    fmt = BridgeReasoningFormatter(
        stage=stage,
        include_bbox=True,
        include_action_tokens=True,
        include_img_next=True,
        thinking_token="<|thinking|>",
        start_token="<|start_of_thinking|>",
        end_token="<|end_of_thinking|>",
        img_next_token="<img_next>",
        img_next_count=16,
        component_order=["SUBTASK", "BBOX", "REASON"],  # 与 bridge 配置一致
    )
    sample = {
        "cot_subtask": "pick up the red block",
        "cot_reasoning": "move to block, close gripper, lift",
        "bbox": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "bbox_valid": True,
        "bbox_confidence": 0.88,
        "action_tokens": "<a1><a2><a3>",
    }
    text = fmt.format("Place the red block on the shelf", sample)
    print(f"\n=== Stage {stage} ===")
    print(text)

if __name__ == "__main__":
    for s in [1, 2, 3, 4]:
        dump(s)