# Bridge-LeRobot 隐式推理数据构建计划

本文档基于现有 `LeRobot` 数据管线与 `ECOT RLDS` 集成的结构，提出将 BRIDGE 数据集扩展为“多阶段 / 隐式推理”格式的详细方案，涵盖标注加载、阶段式格式化、thinking token 对齐、训练适配以及验证流程。

---

## 1. 背景与目标

- **现状**  
  - `starVLA/dataloader/gr00t_lerobot` 已能加载 BRIDGE-LeRobot 数据（action / state / language），并具备空指令过滤、pause frame 删除等能力。  
  - `/share/project/baishuanghao/data/bridge_orig_lerobot/annotations/` 额外提供 `CoT（subtask/reasoning）` 与 `bbox` 标注。  
  - `starVLA/integrations/ecot_rlds` + `training/train_ecot.py` 已支持 `scheduled_stage`+thinking token 的隐式推理策略。  
- **目标**  
  - 在不破坏 LeRobot 原有行为的前提下，为 BRIDGE 数据导入“阶段式 CoT → thinking token”流程，使其与 ECOT RLDS训练策略对齐：  
    1. Stage 0：显式 CoT（文本描述）；  
    2. Stage 1：结构化 tag（便于迁移）；  
    3. Stage 2+：thinking token / latent reasoning（可启用 `forward_latent`）。  
  - 支持逐步替换 `subtask / bbox / move reasoning` 为 latent token，在训练时使用相同的 `enable_latent_reasoning`、`tag2think_count` 等配置。

---

## 2. 模块总体设计

### 2.1 新增 Bridge 注释聚合层

- **文件**：`starVLA/dataloader/gr00t_lerobot/bridge_annotations.py`（建议）  
- **职责**：  
  1. 解析 `episode_dense_captions_full_ep0-20.jsonl` → `step→(subtask, reasoning, gripper_state)`；  
  2. 解析 `episode_sam3_bboxes_final.jsonl` → `step→(bbox, confidence, motion_stats)`；  
  3.（可选）解析 `gripper_triplet_segments.jsonl` → `segment_type / start_step / end_step`；  
  4. 为上层提供 `get_step_annotation(ep, step)`、`get_bbox(ep, step)` 等 API；  
  5. 处理缺失数据：默认返回空字符串、零 bbox、`available=False` mask。

### 2.2 LeRobotSingleDataset 扩展

- **位置**：`starVLA/dataloader/gr00t_lerobot/datasets.py`  
- **增强内容**：  
  1. 初始化时加载 `BridgeAnnotations`（通过配置开关）；  
  2. `get_step_data()` / `__getitem__()` 中追加字段：  
     - `cot_subtask`, `cot_reasoning`, `cot_available`  
     - `bbox_norm`（归一化坐标）、`bbox_conf`, `bbox_valid`  
     -（可选）`segment_type`, `move_stage`  
  3. 若配置 `require_cot` 或 `require_bbox`，在 `_get_all_steps_single_process` 中对缺失 episode 进行过滤；否则保留。  
  4. 所有新字段提供默认值，确保其它数据集不受影响。

### 2.3 Bridge Reasoning Formatter

- **文件**：`starVLA/dataloader/gr00t_lerobot/bridge_reasoning_formatter.py`（建议）  
- **核心功能**：根据配置的 `stage` 生成不同形态的文本/latent：  
  - **Stage 0（显式 CoT）**  
    - 模板：`Instruction + "\nSubtasks:\n1. ...\nReasoning:\n..."`  
    - BBox 用自然语言或 `[BBOX ymin xmin ymax xmax conf=...]` 描述  
  - **Stage 1（结构化 Tag）**  
    - 使用 `[SUBTASK]...[/SUBTASK]`、`[REASON]...[/REASON]`、`[BBOX]...` 包裹，便于后续替换  
  - **Stage 2+（Thinking Token）**  
    - `lang = instruction + " @ " + <|start_of_thinking|> + body + <|end_of_thinking|>`  
    - `body` 中将 tag 内容替换为 `<|thinking|>` 或 `<|SUBTASK|>` 等占位符，根据 `tag2think_count` 控制数量  
  - **输出**：  
    - `formatted_lang`（直接输入模型）  
    - `reasoning_text`（Stage 0/1 用于调试或可选显示）  
    - `thinking_meta`（记录每种 tag 的数量，供日志）

### 2.4 Thinking Token 对齐

- 配置项：  
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
        SUBTASK: 2
        BBOX: 1
        MOVE: 1
  datasets:
    vla_data:
      bridge_reasoning:
        stage: 2
        require_cot: false
        require_bbox: false
  ```  
- Stage 2 及以上必须输出 `<|start_of_thinking|>...<|end_of_thinking|>`，与 RLDS 生成的结构保持一致，以便 `QwenGR00T.forward_latent()` 利用 thinking token 分批迭代。

---

## 3. 实施步骤

### 3.1 阶段 0：显式 CoT 集成
1. **配置**：在 Hydra config 中添加 `bridge_reasoning` 路径、阶段、开关。  
2. **注释加载**：实现 `BridgeAnnotations`，支持 episode→step 查询。  
3. **Dataset 扩展**：`__getitem__` 返回 `cot_subtask` 等字段，Stage=0 时 `lang` = 原始指令 + CoT 文本。  
4. **可视化/调试**：添加 CLI 或 notebook 检查样本输出。

### 3.2 阶段 1：结构化 Tag 化
1. 在 Stage=1 时 `BridgeReasoningFormatter` 输出带 `[SUBTASK]`、`[REASON]` 的文本。  
2. 保持 `build_dataloader` 和训练脚本无须特殊处理，模型输入随文本变化。  
3. 提供 `tag→thinking` 的映射表（供 Stage 2 使用）。

### 3.3 阶段 2：Thinking Token / Latent Reasoning
1. Stage>=2 时，`formatter` 输出 thinking token 包裹文本；将 tag 替换为 `<|thinking|>` 占位符。  
2. `train_ecot.py` 中已存在的 `validate_ecot_config`、`forward_latent` 流程即可复用，确保 dataset `dataset_py="lerobot_datasets"` 也支持 latent reasoning。  
3. 若必要，在 `build_dataloader` 中检测 `stage>=2`→提醒用户必须启用 `framework.enable_latent_reasoning`。  
4. 调整 `tag2think_count` 以设定每个阶段的 token 数（参照 RLDS 的 `"TASK"`、`"PLAN"` 等）。

### 3.4 额外增强（可选）
1. **BBox 数值输入**：提供 `bbox_norm` 向量，可在模型 forward 中作为额外模态（必要时）。  
2. **阶段类型**：若 `gripper_triplet_segments` 可用，可在 CoT 中注明 `move_to_object / grasp_object / ...`。  
3. **日志**：统计每个阶段/segment 的 CoT 覆盖率、bbox 有效率，便于质量评估。

---

## 4. 训练与验证

1. **Stage 0/1**：可使用通用训练脚本（如 `training/train.py`），只需保证 tokenizer 能处理新增 tag。  
2. **Stage 2 及以上**：推荐使用 `training/train_ecot.py`，因为其中已经实现：  
   - `enable_latent_reasoning` 校验；  
   - `forward_latent()` 迭代；  
   - `compute_language_loss` & `vlm_loss_weight`；  
   - 兼容任意 dataset，只要样本 `lang` 中包含 thinking token。  
3. **测试脚本**：参考 `starVLA/integrations/ecot_rlds/tests/`，新增：
   - `tests/bridge_reasoning_contract_test.py`：验证 Stage0/1/2 输出结构；  
   - `tests/bridge_reasoning_smoke.py`：随机抽样，打印 `lang`、thinking token 统计。

---

## 5. 路线图总结

| 阶段 | 目标 | 关键交付 |
| ---- | ---- | -------- |
| Phase 0 | 实现注释解析 + 显式 CoT 输出 | `BridgeAnnotations`, Dataset 扩展, Stage0 formatter |
| Phase 1 | 结构化 tag 化 | tag 模板、Stage1 formatter、样本验证 |
| Phase 2 | 与 ECOT latent reasoning 对齐 | thinking token formatter、`tag2think_count` 对齐、train_ecot 支持 LeRobot |
| 后续 | 增强 bbox/segment 输入、统计、可视化 | 可选任务 |

---

## 6. 参考目录

- 数据注释：`/share/project/baishuanghao/data/bridge_orig_lerobot/annotations/`  
- 注释生成脚本：`/share/project/lvjing/datas/data_pipeline_bridge/data_gen/`  
- 现有 latent reasoning 实现：`starVLA/integrations/ecot_rlds/*`, `training/train_ecot.py`, `model/framework/QwenGR00T.py`, `model/modules/vlm/QWen3.py`

---

通过上述步骤，可让 BRIDGE-LeRobot 数据逐步从显式 CoT 过渡到 Thinking Tokens，与 ECOT RLDS 共享同一套隐式推理训练流程，并为后续多种标注（subtask / bbox / move reasoning）向 latent 表示演进奠定基础。***
