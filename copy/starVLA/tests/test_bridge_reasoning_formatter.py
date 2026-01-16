import numpy as np

from starVLA.dataloader.gr00t_lerobot.bridge_reasoning_formatter import BridgeReasoningFormatter


def _mock_sample():
    return {
        "cot_subtask": "reach toward the spoon",
        "cot_reasoning": "move towards the spoon to grasp it",
        "cot_gripper_state": 1,
        "cot_available": True,
        "bbox": np.array([0.5, 0.1, 0.7, 0.2], dtype=np.float32),
        "bbox_valid": True,
        "bbox_confidence": 0.75,
    }


def test_stage0_formatter_outputs_plain_text():
    formatter = BridgeReasoningFormatter(stage=0, include_bbox=True)
    text = formatter.format("put spoon on tray", _mock_sample())
    assert "put spoon on tray @ Subtask:" in text
    assert "Subtask: reach toward the spoon." in text
    assert "Reasoning: move towards the spoon to grasp it." in text
    assert "BBox:" in text


def test_stage2_formatter_only_subtask_latent():
    formatter = BridgeReasoningFormatter(
        stage=2,
        include_bbox=True,
        thinking_token="<|thinking|>",
        start_token="<|start_of_thinking|>",
        end_token="<|end_of_thinking|>",
        tag2think_count={"SUBTASK": 2, "REASON": 1},
    )
    text = formatter.format("put spoon on tray", _mock_sample())
    assert "put spoon on tray @ <|start_of_thinking|>" in text
    assert "<|end_of_thinking|>" in text
    assert text.count("<|thinking|>") >= 2
    assert "[REASON] move towards the spoon to grasp it [/REASON]" in text
    assert "[BBOX]" in text


def test_stage3_formatter_subtask_and_reason_latent():
    formatter = BridgeReasoningFormatter(stage=3, include_bbox=True)
    text = formatter.format("put spoon on tray", _mock_sample())
    assert "put spoon on tray @ <|start_of_thinking|>" in text
    assert text.count("<|thinking|>") >= 2
    assert "[BBOX]" in text  # bbox still explicit


def test_stage4_formatter_all_latent():
    formatter = BridgeReasoningFormatter(stage=4, include_bbox=True)
    text = formatter.format("put spoon on tray", _mock_sample())
    assert "put spoon on tray @ <|start_of_thinking|>" in text
    assert text.count("<|thinking|>") >= 3
    assert "[BBOX]" not in text


if __name__ == "__main__":
    for stage in range(0, 5):
        formatter = BridgeReasoningFormatter(stage=stage, include_bbox=True)
        text = formatter.format("put the small spoon from basket to tray.", _mock_sample())
        print(f"\n=== Stage {stage} Formatted Text ===")
        print(text)
