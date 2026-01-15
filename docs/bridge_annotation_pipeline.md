# Bridge 数据标注流程（data_pipeline_bridge）

本文档总结 `/share/project/lvjing/datas/data_pipeline_bridge` 中用于 **BRIDGE-LeRobot** 的离线标注流水线，最终产物用于训练侧 `starVLA/dataloader/gr00t_lerobot/bridge_annotations.py` 读取。

## 0. 数据根目录与目标产物

- 数据根目录（LeRobot 格式）：`/share/project/baishuanghao/data/bridge_orig_lerobot`
- 训练侧主要需要两份标注（JSONL）：
  - CoT：`annotations/episode_dense_captions_full_final.jsonl`
  - BBox：`annotations/episode_sam3_bboxes_final_merged.jsonl`

它们会分别在训练 YAML 中作为：

```yaml
datasets:
  vla_data:
    bridge_annotations:
      cot_path:  "/share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_dense_captions_full_final.jsonl"
      bbox_path: "/share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_sam3_bboxes_final_merged.jsonl"
```

---

## 1. 流水线总览（依赖关系）

### 1.1 子流程 A：Gripper 分段（辅助 meta）

作用：基于 state/action 的 gripper 事件（开/合）把 episode 粗分段，后续用于：
- 选择 “pre-grasp / move_to_object / grasp_object / move_to_goal” 的关键帧；
- 作为 CoT 渲染时的分段骨架。

脚本：
- `data_gen/bridge_gripper_triplet_segments.py`

输出：
- `<bridge_root>/intermediate/gripper_triplet_segments.jsonl`

每行包含：
- `episode_index`, `num_steps`, `segmentation_quality`
- `segments[]`: `{segment_type, start_t, end_t}`，`segment_type ∈ {move_to_object, grasp_object, move_to_goal, place_object}`（低质量退化为 `whole_episode`）。

### 1.2 子流程 B：instruction → manipulated_object（Qwen3-VL）

作用：从 `episodes.jsonl` 中抽出每个 episode 的 instruction，并用 **pre-grasp 图片帧**（来自子流程 A）辅助，让 Qwen3-VL 输出被操作物体名 `manipulated_object`。

脚本：
- `data_gen/bridge_instruction_object_extraction.py`

输入：
- `<bridge_root>/meta/episodes.jsonl`
- `<bridge_root>/meta/info.json`（解析视频路径模板）
- `<bridge_root>/intermediate/gripper_triplet_segments.jsonl`（选择 pre-grasp 帧）

输出（默认）：
- `<bridge_root>/annotations/episode_object_prompts_final.jsonl`

每行包含：
- `episode_index`
- `instruction`
- `manipulated_object`（≤3 词、小写、无位置描述）
- `secondary_objects`（可为空）

### 1.3 子流程 C：SAM3 bbox（基于 object prompt 的视频传播 + 轨迹筛选）

作用：用 `manipulated_object` 作为 text prompt，在视频上跑 SAM3，并在多目标 track 中选择“运动最显著”的目标作为 active bbox 轨迹。

脚本：
- `data_gen/bridge_sam3_bbox_from_objects.py`

输入：
- `<bridge_root>/annotations/episode_object_prompts_final.jsonl`
- `<bridge_root>/meta/episodes.jsonl`（提供 episode length）
- `<bridge_root>/intermediate/gripper_triplet_segments.jsonl`（选择 prompt frame：`move_to_object/grasp_object/move_to_goal` 等策略）
- `<bridge_root>/videos/.../episode_XXXXXX.mp4`

输出（默认，且会按 prompt 策略改名）：
- `<bridge_root>/annotations_1/episode_sam3_bboxes_final1_<prompt_strategy>.jsonl`

每行包含：
- `episode_index`, `instruction`, `manipulated_object`, `num_steps`
- `dense_labels.active_bbox`: 长度为 `num_steps` 的 bbox 列表（`[ymin,xmin,ymax,xmax]`，可能为 `null`）
- `dense_scores`: 每步置信度（可为 `null`）
- `step_bboxes`: 稀疏形式的 `{step, bbox_2d, confidence}`
- `motion_stats`, `is_static_warning`

### 1.4 子流程 D：多策略 bbox 合并（merge + 插值补洞）

作用：当你用不同 prompt_strategy 跑出多份 bbox 结果时，做逐帧合并（优先非空/高置信）并对短空洞做插值填充，输出训练侧更稳定的 merged bbox。

脚本：
- `data_gen/merge_sam3_bboxes.py`

输入：
- 一组 bbox JSONL（默认用 glob pattern 匹配，按你的实际文件名调整 `--pattern`）

输出（默认）：
- `<bridge_root>/annotations/episode_sam3_bboxes_final_merged.jsonl`

> 注意：`merge_sam3_bboxes.py` 默认 `--pattern="episode_sam3_bboxes_final_*.jsonl"`，而 `bridge_sam3_bbox_from_objects.py` 默认输出是 `episode_sam3_bboxes_final1_*.jsonl`（多了一个 `1`）。实际使用时建议二选一：
> - 方案 1：运行 merge 时显式传 `--pattern "episode_sam3_bboxes_final1_*.jsonl"`；
> - 方案 2：把 bbox 输出文件重命名到 `episode_sam3_bboxes_final_*.jsonl` 风格。

### 1.5 子流程 E：PG-DC CoT（Physics + Qwen3-VL 的 step 级 subtask/reasoning）

作用：生成训练需要的 step 级 CoT 字段 `subtask/reasoning/gripper_state`，并组织成 `episode_index -> steps{t -> ...}` 的结构。

脚本：
- `data_gen/bridge_pg_dc_prompts.py`

输入：
- `<bridge_root>/intermediate/gripper_triplet_segments.jsonl`（分段骨架）
- `<bridge_root>/annotations/episode_object_prompts.jsonl`（episode → instruction/object prompt）
- `<bridge_root>/data/.../episode_XXXXXX.parquet`（state/action 用于 physics）
- `<bridge_root>/videos/.../episode_XXXXXX.mp4`（关键帧给 Qwen 识别物体/场景）

输出（默认会按 episode 区间分片）：
- `<bridge_root>/annotations/episode_dense_captions_full_ep{start}-{end}.jsonl`（pretty JSON，每个 episode 一段 JSON）

每条 episode 结构：
- `episode_index`
- `total_frames`
- `steps`: `{ "0": {"subtask":..., "reasoning":..., "gripper_state":...}, ... }`

> 注意：该脚本默认读取 `annotations/episode_object_prompts.jsonl`，而子流程 B 默认写 `annotations/episode_object_prompts_final.jsonl`。你可以：
> - 直接把 `_final` 文件拷贝/重命名成 `episode_object_prompts.jsonl`；
> - 或运行时传 `--episode_prompts annotations/episode_object_prompts_final.jsonl`。

### 1.6 子流程 F：CoT 分片合并 + 转标准 JSONL

作用：把分片的 CoT 文件合成一个总文件，并把 pretty/多对象格式转成标准逐行 JSONL（训练侧读取用）。

脚本：
- `data_gen/concat_dense_captions.py`
- `data_gen/convert_dense_captions_to_jsonl.py`

输出：
- `<bridge_root>/annotations/episode_dense_captions_full_combined.jsonl`
- `<bridge_root>/annotations/episode_dense_captions_full_final.jsonl`

---

## 2. 推荐执行顺序（命令示例）

以下命令以 `<bridge_root>=/share/project/baishuanghao/data/bridge_orig_lerobot` 为例，脚本位于：
`/share/project/lvjing/datas/data_pipeline_bridge/data_gen/`。

### 2.1 生成 gripper 分段

```bash
python /share/project/lvjing/datas/data_pipeline_bridge/data_gen/bridge_gripper_triplet_segments.py \
  --bridge_root /share/project/baishuanghao/data/bridge_orig_lerobot \
  --split train \
  --output intermediate/gripper_triplet_segments.jsonl
```

### 2.2 抽取 episode 级 manipulated_object（Qwen3-VL）

```bash
python /share/project/lvjing/datas/data_pipeline_bridge/data_gen/bridge_instruction_object_extraction.py \
  --bridge_root /share/project/baishuanghao/data/bridge_orig_lerobot \
  --output annotations/episode_object_prompts_final.jsonl \
  --triplet_segments_path intermediate/gripper_triplet_segments.jsonl \
  --batch_size 512
```

（可选）让 PG-DC 直接读取该文件：

```bash
cp /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_object_prompts_final.jsonl \
   /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_object_prompts.jsonl
```

### 2.3 运行 SAM3 bbox（多策略可并行）

```bash
python /share/project/lvjing/datas/data_pipeline_bridge/data_gen/bridge_sam3_bbox_from_objects.py \
  --bridge_root /share/project/baishuanghao/data/bridge_orig_lerobot \
  --episode_prompts annotations/episode_object_prompts_final.jsonl \
  --gripper_segments intermediate/gripper_triplet_segments.jsonl \
  --output annotations_1/episode_sam3_bboxes_final1.jsonl \
  --prompt_strategy move_to_object \
  --sam3_checkpoint <PATH_TO_SAM3_CKPT> \
  --sam3_gpus 0
```

你可以把 `--prompt_strategy` 换成 `first/last/grasp_object/move_to_goal`，生成多份 bbox 结果。

### 2.4 合并多策略 bbox → merged

```bash
python /share/project/lvjing/datas/data_pipeline_bridge/data_gen/merge_sam3_bboxes.py \
  --bridge_root /share/project/baishuanghao/data/bridge_orig_lerobot \
  --input_dir annotations_1 \
  --pattern "episode_sam3_bboxes_final1_*.jsonl" \
  --output annotations/episode_sam3_bboxes_final_merged.jsonl
```

### 2.5 生成 CoT（PG-DC，按区间分片）

```bash
python /share/project/lvjing/datas/data_pipeline_bridge/data_gen/bridge_pg_dc_prompts.py \
  --bridge_root /share/project/baishuanghao/data/bridge_orig_lerobot \
  --triplet_segments intermediate/gripper_triplet_segments.jsonl \
  --episode_prompts annotations/episode_object_prompts_final.jsonl \
  --output annotations/episode_dense_captions_full.jsonl \
  --ep_start 0 \
  --ep_end 999 \
  --pretty
```

分片输出会变成：`annotations/episode_dense_captions_full_ep0-999.jsonl`。可按区间并行跑。

### 2.6 合并 CoT 分片并转 final JSONL

```bash
python /share/project/lvjing/datas/data_pipeline_bridge/data_gen/concat_dense_captions.py \
  --annotations_dir /share/project/baishuanghao/data/bridge_orig_lerobot/annotations \
  --pattern "episode_dense_captions_full_ep*-*.jsonl" \
  --output /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_dense_captions_full_combined.jsonl

python /share/project/lvjing/datas/data_pipeline_bridge/data_gen/convert_dense_captions_to_jsonl.py \
  --input /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_dense_captions_full_combined.jsonl \
  --output /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/episode_dense_captions_full_final.jsonl
```

---

## 3. 最终产物字段契约（训练侧使用）

### 3.1 CoT 文件：`episode_dense_captions_full_final.jsonl`

每行（episode 级）：

```json
{
  "episode_index": 0,
  "total_frames": 26,
  "steps": {
    "0": {"subtask": "...", "reasoning": "...", "gripper_state": 1},
    "1": {"subtask": "...", "reasoning": "...", "gripper_state": 1}
  }
}
```

训练侧读取：按 `(episode_index, step_index)` 查 `steps[str(step_index)]`。

### 3.2 BBox 文件：`episode_sam3_bboxes_final_merged.jsonl`

每行（episode 级）：

```json
{
  "episode_index": 0,
  "num_steps": 26,
  "dense_labels": {"active_bbox": [null, [ymin,xmin,ymax,xmax], ...]},
  "dense_scores": [null, 0.91, ...],
  "step_bboxes": [{"step": 1, "bbox_2d": [...], "confidence": 0.91}]
}
```

训练侧读取：按 `(episode_index, step_index)` 查 `dense_labels.active_bbox[step_index]`（若为 `null` 则视为缺失）。

---

## 4. 常见注意事项

- `episode_object_prompts_final.jsonl` vs `episode_object_prompts.jsonl`：建议统一文件名或用 CLI 参数显式指定，避免 PG-DC 找不到 prompts。
- bbox 文件命名 pattern：`merge_sam3_bboxes.py` 的 `--pattern` 需要与实际输出文件名匹配（是否带 `final1`）。
- bbox 坐标格式：下游 `BridgeAnnotations` 会把 `dense_labels.active_bbox` 直接 `np.asarray`；请保证每个 bbox 为长度 4 的数值数组，且最好统一为归一化 `xyxy`（或在下游明确处理像素坐标/归一化）。
- 分片并行：CoT 生成建议用 `--ep_start/--ep_end` 切分并行跑，最后用 concat + convert 汇总到 final。

