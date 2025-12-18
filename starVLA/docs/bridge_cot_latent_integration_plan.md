# Bridge-LeRobot CoT 标注与隐式推理一体化方案

> 本文是对 `cot_integration_plan.md` 和 `bridge_latent_reasoning_plan.md` 的统一与细化，后续所有实现以本文件为总蓝图。

---

## 1. 总体目标

我们希望在 **不破坏现有 LeRobot 数据管线** 的前提下，为 BRIDGE-LeRobot 数据集加入：

1. **显式 CoT 标注集成**  
   - 利用已有的 `subtask` / `reasoning` / `bbox` 标注文件，将其对齐到 LeRobot 的 `(trajectory_id, step)` 上，并成为样本的一部分。
2. **隐式推理（latent reasoning）训练能力**  
   - 参考 `ECOT RLDS` 集成（`starVLA/integrations/ecot_rlds` 与 `training/train_ecot.py`），为 BRIDGE 数据构建分阶段的推理表示：  
     - Stage 0：纯文本显式推理；  
     - Stage 1：带结构化 tag 的显式推理；  
     - Stage 2+：使用 thinking tokens (`<|thinking|>` 等)，可启用 `forward_latent()` 进行隐式推理训练。

**硬约束：**

- 保留 LeRobot 的所有过滤机制与统计逻辑：  
  - 空指令 episode 过滤（`_get_all_steps_single_process` + `get_language`）；  
  - `delete_pause_frame` 删除静止帧；  
  - Mixture 采样 / 统计导出等逻辑不被破坏。
- 对其它非 BRIDGE 数据集的行为零侵入。

---

## 2. 现有系统概览

### 2.1 LeRobot 数据管线（当前）

- 入口：`starVLA/dataloader/__init__.py: build_dataloader`  
  - `dataset_py="lerobot_datasets"` 时：  
    - `starVLA/dataloader/lerobot_datasets.py:get_vla_dataset`  
    - 使用 `gr00t_lerobot` 包：  
      - `LeRobotSingleDataset` / `LeRobotMixtureDataset`（`gr00t_lerobot/datasets.py`）  
      - `ROBOT_TYPE_CONFIG_MAP` + `DATASET_NAMED_MIXTURES`（`data_config.py`, `mixtures.py`）
- `LeRobotSingleDataset.__getitem__` 当前输出：  
  - `image`: `[PIL.Image, ...]`（已裁剪&resize）  
  - `language`: `str`（来自 `annotation.human.action.task_description`）  
  - `action`: `np.ndarray`（concat 状态动作维度）  
  - 无显式 CoT 与 bbox。

### 2.2 Bridge 标注文件t

位于 `/share/project/baishuanghao/data/bridge_orig_lerobot/`：

- `meta/*`：  
  - `episodes.jsonl`: 每个 episode 的 `episode_index`, `tasks`, `length`；  
  - `tasks.jsonl`: `task_index → 任务文本`；  
  - `modality.json`: state/action/video/annotation 映射；  
  - `steps_*.pkl`: 已计算好的 step 缓存（被 `LeRobotSingleDataset` 使用）。
- `annotations/episode_dense_captions_full_ep0-20.jsonl`：  
  - 每个 episode：`episode_index`, `total_frames`, `steps`；  
  - `steps["t"]` 包含：`subtask`, `reasoning`, `gripper_state`。
- `annotations/episode_sam3_bboxes_final.jsonl`：  
  - 每个 episode：`instruction`, `manipulated_object`, `num_steps`；  
  - `dense_labels.active_bbox[t]`（可能 null）；  
  - `step_bboxes`: `{"step": t, "bbox_2d": [...], "confidence": ...}`；  
  - `motion_stats`, `is_static_warning` 等。

### 2.3 ECOT RLDS 与 latent reasoning（对齐目标）

- ECOT RLDS 数据集适配：`starVLA/integrations/ecot_rlds/*`  
  - `ECOTRLDSDataset` 使用 Prismatic RLDS 提供：  
    - `image`, `lang`, `action`, `state`  
    - `reasoning`（包含 thinking tokens）、`dataset_name` 等。  
  - `ECOTBatchTransform` 将 RLDS batch 转为 StarVLA 样本。  
  - `config.validate_and_normalize_cfg` 支持 `scheduled_stage`、`reasoning_json`、thinking token 配置。
- 训练脚本：`training/train_ecot.py`  
  - 校验 `framework.enable_latent_reasoning` 与 `framework.latent_reasoning.*`；  
  - Stage 0：普通 forward；Stage ≥1：`QwenGR00T.forward_latent()`；  
  - Qwen 模块：`model/modules/vlm/QWen3.py` 中实现 thinking token 对齐与 span masking。

**我们的目标**：让 BRIDGE-LeRobot 数据通过 LeRobot 管线也能产生与 ECOT RLDS 一致风格的 `lang + reasoning + thinking tokens`，以便共享 `train_ecot.py` 的隐式推理训练流程。

---

## 3. 阶段式目标与里程碑

### 3.1 阶段划分

1. **Phase 0：CoT 标注集成（显式文本）**  
   - 从 annotation JSONL 读取 `subtask`, `reasoning`, `bbox`；  
   - 在 `__getitem__` 样本中返回这些字段；  
   - Stage=0 时，`lang` 仍以显式自然语言形式包含 CoT 内容。
2. **Phase 1：结构化 Tag 化**  
   - 在文本中引入 `[SUBTASK]...[/SUBTASK]`, `[REASON]...[/REASON]`, `[BBOX]...`；  
   - 不改变训练脚本，仅让模型先适应结构化 CoT。
3. **Phase 2：Thinking Token / 隐式推理**  
   - 引入 `<|thinking|>`, `<|start_of_thinking|>`, `<|end_of_thinking|>`；  
   - 使用 `tag2think_count` 控制 latent token 数量；  
   - 适配 `train_ecot.py` 的 latent reasoning 流程。

### 3.2 每阶段输出形式

- Stage 0：  
  - `lang = instruction + "\nSubtasks:\n1. ...\nReasoning:\n..."`  
  - 无 thinking token。
- Stage 1：  
  - `lang = instruction + "\n[SUBTASK]...[/SUBTASK]\n[REASON]...[/REASON]\n[BBOX]..."`
- Stage 2+：  
  - `lang = instruction + " @ " + "<|start_of_thinking|>" + body + "<|end_of_thinking|>"`  
  - `body` 中大部分细节被 `<|thinking|>` 占位，但可以在其他模态/标签中保留结构信息，用于 `forward_latent` 对齐。

---

## 4. 数据层集成：Bridge 注释与 LeRobot 数据集

### 4.1 BridgeAnnotations 模块

**新文件建议**：`starVLA/dataloader/gr00t_lerobot/bridge_annotations.py`

**职责：**

1. **解析 CoT 文件**（`episode_dense_captions_full_ep0-20.jsonl`）：  
   - 建立：  
     ```python
     cot_annotations: Dict[int, Dict[int, Dict[str, Any]]]
     # episode_index -> step_index -> {"subtask": str, "reasoning": str, "gripper_state": int}
     ```
2. **解析 BBox 文件**（`episode_sam3_bboxes_final.jsonl`）：  
   - 建立：  
     ```python
     bbox_annotations: Dict[int, Dict[int, Dict[str, Any]]]
     # episode_index -> step_index -> {"bbox": np.ndarray or None, "conf": float, "valid": bool}
     ```
   - 同时保存 `dense_labels`, `motion_stats` 供后续使用（如统计/筛选）。
3. **封装访问接口**：
   ```python
   def get_step_cot(self, ep: int, step: int) -> dict | None
   def get_step_bbox(self, ep: int, step: int) -> dict | None
   def has_cot(self, ep: int) -> bool
   def has_bbox(self, ep: int) -> bool
   ```
4. **健壮性处理**：  
   - `episode_index` 覆盖不完整时，返回 None；  
   - `step` 超界 / 缺失时，返回 None；  
   - 可记录 `num_cot_missing`, `num_bbox_missing` 用于日志和评估。

### 4.2 与 LeRobotSingleDataset 对接（含过滤机制）

**修改处**：`starVLA/dataloader/gr00t_lerobot/datasets.py: LeRobotSingleDataset`

1. **在 `__init__` 中挂载 BridgeAnnotations（针对 BRIDGE 数据集）**：  
   - 通过配置判断当前数据集是否为 bridge：  
     - `data_mix` in `{"bridge", "bridge_rt_1", ...}`；  
     - 或通过 `dataset_path` / `robot_type` 判断。  
   - 若开启 `bridge_annotations`，则在 dataset 实例上保存 `self.bridge_annotations`。
2. **在 `_get_all_steps_single_process` 中扩展过滤（可选）**：  
   新增可选过滤开关，默认均为 `False`，只对 BRIDGE 数据生效：  
   - **Episode 级过滤**  
     - `require_cot_episode: bool`：  
       - 为 True 时，调用 `self.bridge_annotations.has_cot(trajectory_id)`；  
       - 若该 episode 完全没有 CoT 标注，则整条 trajectory 跳过（类似“空语言”过滤）。  
     - `require_bbox_episode: bool` 或 `min_episode_bbox_coverage: float`：  
       - 例如统计 episode 内有 bbox 的 step 比例 `< min_episode_bbox_coverage` 时跳过整条 episode。  
   - **Step / 帧级过滤**  
     - `require_bbox_step: bool`：  
       - 在为每个 `base_index` 决定是否加入 `all_steps` 时，除现有 `delete_pause_frame` 条件外，再检查 `get_step_bbox`；  
       - 如果该 step 没有有效 bbox（None 或 `bbox_valid=False`），则不加入 `all_steps`。  
   - 所有过滤逻辑仅在 BRIDGE 数据集且 `bridge_annotations` 有效时启用；其它数据集不受影响。
3. **在 `get_step_data` / `__getitem__` 中注入字段**：  
   ```python
   cot = self.bridge_annotations.get_step_cot(trajectory_id, base_index) if self.bridge_annotations else None
   bbox = self.bridge_annotations.get_step_bbox(trajectory_id, base_index) if self.bridge_annotations else None

   data["cot_subtask"] = cot["subtask"] if cot else ""
   data["cot_reasoning"] = cot["reasoning"] if cot else ""
   data["cot_gripper_state"] = cot["gripper_state"] if cot else -1
   data["cot_available"] = cot is not None

   # bbox 归一化到 [0,1] 或 [0,224], 与 image resize 保持一致
   data["bbox"] = bbox["bbox"] if bbox and bbox["bbox"] is not None else np.zeros(4, dtype=np.float32)
   data["bbox_confidence"] = bbox["conf"] if bbox else 0.0
   data["bbox_valid"] = bbox is not None and bbox["bbox"] is not None
   ```
4. **LeRobotMixtureDataset 不需改动**：  
   - 它只透传 dataset 返回的 dict；  
   - 新字段不影响采样逻辑，也不会干扰已有 transforms（主要作用于 video/state/action）。

---

## 5. 文本与 latent 构建：BridgeReasoningFormatter

### 5.1 模块设计

**新文件建议**：`starVLA/dataloader/gr00t_lerobot/bridge_reasoning_formatter.py`

**核心接口：**

```python
class BridgeReasoningFormatter:
    def __init__(self, stage: int, tag2think_count: dict, thinking_tokens_cfg: dict):
        ...

    def format(
        self,
        instruction: str,
        episode_id: int,
        step_id: int,
        sample: dict,
    ) -> dict:
        """
        返回:
          {
            "lang": str,              # 最终喂给 tokenizer 的文本
            "reasoning_text": str,    # 方便调试的完整 CoT 文本（可选）
            "thinking_meta": dict,    # thinking token 统计
          }
        """
```

### 5.2 Stage 0：显式 CoT

- 模板举例：

```text
Instruction: put small spoon from basket to tray.

Step 5:
Subtask: reach toward the small spoon in the basket.
Reasoning: globally the gripper should progress towards the lower direction...
BBox: [0.52, 0.09, 0.70, 0.23] (conf=0.58)
```

- 特性：  
  - 完整暴露 `cot_subtask` / `cot_reasoning`；  
  - bbox 可简要描述或直接以数值形式呈现。

### 5.3 Stage 1：结构化 tag

- 模板举例：

```text
Instruction: put small spoon from basket to tray.

[SUBTASK] reach toward the small spoon in the basket [/SUBTASK]
[REASON] the gripper moves towards the spoon to prepare for grasping [/REASON]
[BBOX] 0.52 0.09 0.70 0.23 (conf=0.58) [/BBOX]
```

- 特性：  
  - Tag 化后，方便将来直接映射为 `<|thinking|>`；  
  - 依然是完全可读的显式 CoT。

### 5.4 Stage 2+：Thinking Tokens / 隐式推理

- 模板举例：

```text
put small spoon from basket to tray @ <|start_of_thinking|> <|thinking|> <|thinking|> <|thinking|> <|end_of_thinking|>
```

- 处理逻辑：  
  1. 根据 `tag2think_count`（如：`{"SUBTASK": 2, "BBOX": 1, "MOVE": 1}`）确定 latent token 总数；  
  2. 不再在 `lang` 中显式展示 CoT 文本，仅保留 thinking tokens；  
  3. 若需要保留监督，可以通过 label mask 让 `<|thinking|>` 区段不计算 loss（参考 ECOT 的做法）。

---

## 6. 配置方案与开关设计

### 6.1 数据集配置

在原有 `datasets.vla_data` 下新增 bridge 专用字段：

```yaml
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_mix: bridge
    bridge_annotations:
      cot_path: /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_dense_captions_full_ep0-20.jsonl
      bbox_path: /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_sam3_bboxes_final.jsonl
      require_cot: false
      require_bbox: false
    bridge_reasoning:
      stage: 0    # 0/1/2
      enable: true
```

### 6.2 框架/模型配置

与 ECOT 保持一致：

```yaml
framework:
  enable_latent_reasoning: true
  latent_reasoning:
    compute_language_loss: true
    vlm_loss_weight: 0.1
    thinking_token: "<|thinking|>"
    start_of_thinking_token: "<|start_of_thinking|>"
    end_of_thinking_token: "<|end_of_thinking|>"
    tag2think_count:
      TASK: 1
      SUBTASK: 1
      MOVE: 1
      BBOX: 1
```

当 `bridge_reasoning.stage >= 2` 时，需要在日志中打印出上述配置，并确保 tokenizer 注册了对应的 special tokens。

---

## 7. 训练脚本适配策略

### 7.1 Stage 0/1：常规训练

- 可以使用现有通用训练脚本（如 `training/train_*.py`）：  
  - DataLoader 输出的新字段由模型前向前的 prompt 构建逻辑使用；  
  - 不需要 `forward_latent`，直接 `forward` 即可。

### 7.2 Stage 2+：隐式推理训练

- 推荐使用 `training/train_ecot.py`，理由：  
  - 内部已经实现：  
    - `validate_ecot_config`（检查 enable_latent_reasoning / latent_reasoning）  
    - 对 `QwenGR00T` 调用 `forward_latent` 的路径；  
    - 对 `thinking_token` / span mask 的处理。  
  - 对于 BRIDGE-LeRobot：  
    - 只要 DataLoader 提供的 `lang` 中包含 `<|start_of_thinking|> ... <|end_of_thinking|>`，`forward_latent` 就可以工作；  
    - 不要求数据一定来自 RLDS。

### 7.3 细节注意点

- Stage 2 以上时：  
  - 保证 `latent_reasoning.thinking_token` 等与 QwenTokenizer 中的 special tokens 一致；  
  - 保证 `tag2think_count` 中的各 tag 数量与我们在 Bridge formatter 中生成的 thinking tokens 总数一致（否则会导致批内对齐失败或无 thinking token）。  
  - 如果仍需对 post-thinking 段进行语言监督，需参照 ECOT `_find_ecot_spans` 的策略，适配 mask。
- 提供了专用配置样例：`config/training/bridge_ecot_stage2.yaml`，直接指向 `dataset_py=lerobot_datasets` 并开启 `bridge_reasoning.stage=2`；调用方式为  
  ```bash
  python starVLA/training/train_ecot.py --config_yaml config/training/bridge_ecot_stage2.yaml
  ```
  可根据需要调整 `stage`/`tag2think_count` 或 batch 大小。

---

## 8. 测试与验证计划

### 8.1 单元测试

1. **BridgeAnnotations**：  
   - 随机抽取几个 episode，检查 `get_step_cot` / `get_step_bbox` 输出；  
   - 覆盖各种情况：完整标注 / 部分缺失 / 全缺失。
2. **LeRobotSingleDataset 扩展**：  
   - 对于 BRIDGE 数据集，检查 `__getitem__` 是否包含新字段；  
   - 确认非 BRIDGE 数据集不受影响。

### 8.2 合同测试（Contract）

参照 `starVLA/integrations/ecot_rlds/tests/dataset_contract_test.py` 新增：

- `tests/bridge_reasoning_contract_test.py`：  
  - 对 Stage 0/1/2 分别构造小配置；  
  - 检查样本中：  
    - `lang` 是否符合预期格式（tag / thinking token）；  
    - `cot_*` 与 `bbox_*` 是否正确对齐；  
    - thinking token 数量是否与 `tag2think_count` 一致。

### 8.3 冒烟测试（Smoke）

新增脚本：

```bash
python -m starVLA.tests.bridge_reasoning_smoke --stage 2 --num_samples 3
```

输出若干样本的 `lang` / bbox / CoT 信息，以便人工检查。

---

## 9. 任务拆解与实施顺序

**Phase 0：CoT 集成**
1. 实现 `BridgeAnnotations` 并接入 BRIDGE 数据集初始化；  
2. 在 `LeRobotSingleDataset.__getitem__` 中加入 `cot_*` / `bbox_*` 字段；  
3. 增加基础单元测试和简单可视化脚本。

**Phase 1：Tag 化**
4. 实现 `BridgeReasoningFormatter` 的 Stage 0/1；  
5. 修改 Dataset，使其在 `__getitem__` 时调用 formatter 生成新的 `lang`；  
6. 添加 contract 测试，确保 tag 格式稳定。

**Phase 2：Thinking Tokens / 隐式推理**
7. 扩展 formatter 的 Stage 2+，输出 thinking tokens；  
8. 配置 `latent_reasoning`，适配 `train_ecot.py`；  
9. 验证 `forward_latent` 能正常工作（包括 thinking span mask）；  
10. 针对 Bridge 数据运行端到端的短训练，检查 loss 曲线与日志行为。

**后续增强（Optional）**
11. 将 bbox 作为额外模态输入（如 ROI cropping 或数值 embedding）；  
12. 引入 `segment_type`（move_to_object 等）到 prompt/latent 中，进一步丰富策略表达；  
13. 完善统计导出，使 `dataset_statistics.json` 记录动作/状态之外的辅助信息。

---

## 10. 总结

通过以上统一方案，我们可以：

- 将 BRIDGE-LeRobot 的 `subtask` / `reasoning` / `bbox` 标注以结构化方式集成到 LeRobot 数据管线；  
- 利用阶段式 formatter 和 thinking tokens，将 CoT 从显式文本平滑过渡到 latent 表示；  
- 复用 ECOT RLDS 已有的隐式推理训练框架，实现机器人任务上的更强推理能力；  
- 同时确保对现有数据集与训练脚本保持良好兼容性。
