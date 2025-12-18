# Bridge 数据集 CoT/BBox 标注融合计划

本文档梳理如何把 `/share/project/baishuanghao/data/bridge_orig_lerobot/annotations/` 下的 **CoT 字幕**（`episode_dense_captions_full_ep0-20.jsonl`）与 **SAM3 目标框**（`episode_sam3_bboxes_final.jsonl`）融合进现有 `LeRobotSingleDataset` / `LeRobotMixtureDataset` 数据管线，同时保持当前的过滤逻辑、统计与下游接口兼容。

---

## 1. 目标 & 约束

- **保留现有行为**  
  - 空指令过滤（`_get_all_steps_single_process` 依赖 language modality 判空）。  
  - `delete_pause_frame` 机制、重试逻辑、统计导出等。
- **新增信息**  
  - 每一步的 `subtask` / `reasoning`（来自 dense caption 文件）。  
  - 每一步的 `bbox`（含置信度 / null 情况）用于视觉监督。
- **兼容性**  
  - 未配置额外标注的其它数据集不能受到影响。  
  - 默认 `collate_fn`（identity）与 downstream 代码无需立刻修改即可继续运行。
- **可扩展**  
  - 允许未来扩展更多标注（如多物体 bbox、阶段标签）而不破坏接口。

---

## 2. 数据源概览

| 文件 | 结构摘要 | 关键字段 |
| ---- | -------- | -------- |
| `episode_dense_captions_full_ep0-20.jsonl` | 按 episode 存储 `{episode_index, total_frames, steps}`；`steps` 是以字符串索引的字典 | `steps["0"].subtask`, `steps["0"].reasoning`, `steps["0"].gripper_state` |
| `episode_sam3_bboxes_final.jsonl` | 按 episode 存储 bbox | `dense_labels.active_bbox`（list，含 null），`step_bboxes`（带 `step`, `bbox_2d`, `confidence`） |

注意：目前 CoT 文件仅覆盖部分 episode（0-20 等）。bbox 文件覆盖范围更大但仍需与 `trajectory_length` 对齐。缺失数据需要 graceful fallback。

---

## 3. 配置扩展方案

1. **新增数据集配置字段**（优先放在 Hydra/OmegaConf 的 `cfg.datasets.vla_data` 中，以同一套配置控制多个数据集）：  
   ```yaml
   datasets:
     vla_data:
       custom_annotations:
         bridge:
           cot_path: /share/.../episode_dense_captions_full_ep0-20.jsonl
           bbox_path: /share/.../episode_sam3_bboxes_final.jsonl
           require_cot: false   # true = 没有 CoT 的 episode 直接过滤
           bbox_mode: step_bboxes  # 或 dense_labels
   ```
2. **在 `get_vla_dataset` 里透传**这些路径给 `LeRobotSingleDataset`（或更细粒度的 `LeRobotMixtureDataset`），避免在 `ROBOT_TYPE_CONFIG_MAP` 中写死。
3. **确保默认值为 None**：当配置为空或路径不存在时，现有逻辑完全不执行，保证其他机器人数据集无感知。

---

## 4. 数据加载与缓存

1. **解析 JSONL**（`LeRobotSingleDataset.__init__`）：
   - 读取一次，构建 `Dict[int, EpisodeAnnotations]`。  
   - 对于 CoT：`EpisodeCoT = {step_index(int): {"subtask": str, "reasoning": str, "gripper_state": int}}`。  
   - 对于 bbox：  
     - `dense_labels.active_bbox` 可直接视为 `List[Optional[List[float]]]`。  
     - `step_bboxes` 更明确，解析为 `Dict[int, {"bbox": Optional[np.ndarray], "confidence": float}]`。  
   - 缓存在 dataset 实例上，避免重复 IO。

2. **健壮性校验**：  
   - 若 `total_frames` 与 `trajectory_length` 不一致，打印警告并对齐为最短长度。  
   - 若 `bbox`/`subtask` 缺某些 step，后续访问时返回占位（空字符串、零框）和 mask。  
   - 使用 `require_cot`/`require_bbox` 配置决定是否因此跳过整个 episode。

---

## 5. 数据过滤策略

### 5.1 继承现有过滤

- `_get_all_steps_single_process` 里已有的语言判空、`delete_pause_frame` 等逻辑保持不变。

### 5.2 新增过滤（可选）

- 若配置 `require_cot=true`，则在 `_get_all_steps_single_process` 检查 `episode_annotations.has_cot(trajectory_id)`。若不存在则跳过该 trajectory。  
- 同理可扩展 `require_bbox`。

### 5.3 记录统计

- 在日志中记录：  
  - `num_cot_missing`, `num_bbox_missing`。  
  - 方便排查注释覆盖率。

---

## 6. 样本增强与返回格式

### 6.1 `get_step_data`（或 `__getitem__`）增强

1. 已有步骤：获取 `images`、`action`、`language`。  
2. **新增字段**（示例）：
   ```python
   step_annotations = self.cot_annotations.get(trajectory_id, {}).get(base_index)
   bbox_info = self.bbox_annotations.get(trajectory_id, {}).get(base_index)

   data["cot_subtask"] = step_annotations.get("subtask", "")
   data["cot_reasoning"] = step_annotations.get("reasoning", "")
   data["cot_available"] = step_annotations is not None

   data["bbox"] = bbox_info.get("bbox", np.zeros(4, dtype=np.float32))
   data["bbox_confidence"] = bbox_info.get("confidence", 0.0)
   data["bbox_valid"] = bbox_info.get("bbox") is not None
   ```
3. 若将来需要多目标 bbox，可把 `bbox` 改成 `(num_objects, 4)` 并附 mask。

### 6.2 混合数据集传递

- `LeRobotMixtureDataset.__getitem__` 返回 `dict(action=..., image=..., lang=..., **extra)`，保持向下兼容。  
- 当前默认 `collate_fn` 是 identity，因此 DataLoader 会输出 list，需要训练脚本自行处理新字段（比如拼 string 或喂多头网络）。

---

## 7. 下游使用建议

1. **文本模型**：可以把 `language`, `cot_subtask`, `cot_reasoning` 拼成更复杂的 prompt，例如：  
   ```
   指令: {language}
   当前子任务: {cot_subtask}
   推理: {cot_reasoning}
   ```
2. **视觉模型**：若需要可视化 bbox，需确保图像 resize 与 bbox坐标一致（annotation 是归一化坐标，适合直接乘以 224）。  
3. **训练脚本**：更新 collate / batch 处理逻辑以支持新的字段（比如把字符串 list 合成 prompt、把 bbox list 转 tensor）。这部分在 dataloader 之外，按需实现。

---

## 8. 实施步骤列表

1. **配置层**：更新 Hydra 配置以注入路径、策略。  
2. **Dataset 初始化**：读取 JSONL、缓存，添加缺失统计。  
3. **过滤逻辑**：可选地根据配置跳过缺少注释的 episode。  
4. **样本增强**：在 `get_step_data`/`__getitem__` 添加 CoT / bbox 字段和 mask。  
5. **回归验证**：  
   - 无注释配置下数据流应与当前完全一致。  
   - 注释存在时检查随机样本输出，确保字段对齐。  
6. **下游脚本同步**：更新训练脚本读取新字段，添加 prompt / loss 支持。

---

## 9. 后续扩展

- 若未来想支持多数据源（例如多个 JSONL），可在配置里改为列表，并在解析阶段 merge。  
- 可以加入缓存文件（例如 `.pkl`）加速加载，类似当前的 `steps_*.pkl`。  
- 统计模块（`save_dataset_statistics`）如需记录 bbox 分布，可在未来扩展。

---

## 10. 参考

- 数据生成脚本：`/share/project/lvjing/datas/data_pipeline_bridge/data_gen`。  
- 关键代码位置：
  - `starVLA/dataloader/lerobot_datasets.py` – 数据集构造。  
  - `starVLA/dataloader/gr00t_lerobot/datasets.py` – 数据读取 / 过滤 / 采样。  
  - `starVLA/dataloader/gr00t_lerobot/data_config.py` – 模态声明。  
- 原始元数据：`/share/project/baishuanghao/data/bridge_orig_lerobot/meta/`。

---

以上计划为后续实现提供路线图，确保 CoT / bbox 标注在不破坏既有数据流程的前提下顺利融合。下一步即可按步骤逐项落地。
