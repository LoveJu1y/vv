# BRIDGE 指令驱动的 SAM3 BBox 标注方案（草案）

> 目标：对 **有效 episode** 的视频进行自动目标框标注  
> 信号来源：`bridge_orig_lerobot` 中的自然语言指令（`tasks.jsonl` / `episodes.jsonl`）  
> 工具：大语言模型（LLM/VLM） + SAM3 BBox 检测 + 现有几何分段结果

---

## 1. 场景与现有资产梳理

- 原始数据集：`/share/project/baishuanghao/data/bridge_orig_lerobot`
  - `meta/episodes.jsonl`：`episode_index` ↔ `task_index`
  - `meta/tasks.jsonl`：`task_index` ↔ `task`（自然语言指令）
  - `meta/info.json`：`video_path` 模板、`chunks_size` 等
  - `data/chunk-xxx/episode_xxxxxx.parquet`：低维 state/action
  - `videos/.../episode_xxxxxx.mp4`：多机位视频
- 有效 episode 概念：
  - 在 `steps_2d5a34b904d2.pkl` 中已经过滤掉：
    - instruction 为空的 episode；
    - 以及部分“全程静止/无有效动作”的 episode（`delete_pause_frame=True`）。
  - 对于离线标注，我们可以 **进一步只处理有非空 instruction 的 episode**。
- 已有几何分段与可视化脚本（在 `data_pipeline_bridge` 仓库）：
  - `bridge_geometric_segmentation.py` / `bridge_segmentation.py`：
    - 读取 BRIDGE 数据，输出 `geo_segments.jsonl`，包含：
      - `episode_index`
      - `num_steps`
      - `cycles[*].segments[*]`（`segment_type, start_t, end_t`）
      - `dense_labels.subtask_id / cycle_id`
  - `visualize_segmentation.py`：可视化几何分段结果。
- 已有 SAM3 BBox 集成（桥接仓库）：
  - `data_gen/bridge_sam3_bbox.py`：
    - 从 `geo_segments.jsonl` 读取 segment；
    - 直接用 **整条 instruction 或 default_prompt** 作为 SAM3 文本 prompt；
    - 调用 `Sam3Detector.detect(...)` 得到 bbox，填入：
      - `segment["grounding"]["bbox_2d"]`
      - `dense_labels.active_bbox[t]`。
- 当前缺口：
  - 没有一步 **通过 LLM 从 instruction 中抽取“被操作物体名称”** 的过程；
  - SAM3 的 prompt 质量有限（直接用整句指令，噪声多、歧义大）。

---

## 2. 总体方案概览

整体拆成三条离线流水线阶段，每一阶段都写入中间结果，便于复用与调试：

1. **Episode 级指令解析（LLM）**  
   - 输入：`meta/tasks.jsonl` / `meta/episodes.jsonl`  
   - 去重：按 `task` 文本做去重（同一指令多 episode 复用结果）  
   - 调用 LLM，抽取：
     - `manipulated_object`: 主要被操作物体（短语，如 `"green sponge"`）
     - 可选：`secondary_objects`: 其他提及物体  
   - 输出：`annotations/task_object_prompts.jsonl`
     - 每行：`{"task_index": int, "task": str, "manipulated_object": str, "secondary_objects": [str], "llm_raw": str}`

2. **Episode 级有效性过滤 + 映射**  
   - 输入：
     - `meta/episodes.jsonl`
     - `annotations/task_object_prompts.jsonl`
     - （可选）`steps_2d5a34b904d2.pkl` / `geo_segments.jsonl`
   - 逻辑：
     - 仅保留：`task` 非空、`manipulated_object` 非空的 episode；
     - 可选再 intersect：只保留同时出现在几何分段输出 / steps PKL 中的 episode；
   - 输出：一个映射表 `annotations/episode_object_prompts.jsonl`：
     - `{"episode_index": int, "task_index": int, "instruction": str, "manipulated_object": str}`

3. **基于物体名称的 SAM3 BBox 标注**  
   - 在已有的 `bridge_sam3_bbox.py` 基础上，新增/调整一版脚本：
     - 优先使用 `episode_object_prompts.jsonl` 中的 `manipulated_object` 作为 SAM3 prompt；
     - 对每个 segment 选择关键帧，运行 SAM3；
     - 写回 `grounding.bbox_2d`，并构建稠密 `dense_labels.active_bbox`。
   - 输出：`annotations/segmentation_with_vlm_object_sam3.jsonl`

整个过程完全离线，不影响训练 dataloader；后续训练只需要读取新的 JSONL 注释即可。

---

## 3. 阶段一：指令 → 物体名称（LLM 抽取）

### 3.1 输入与去重策略

- 读取：
  - `meta/tasks.jsonl`
  - 每行格式大致为：`{"task_index": int, "task": str, ...}`
- 去重：
  - 用 `task` 文本作为 key（strip + 规范化空白），构建：
    - `unique_tasks: Dict[str, int]`（或同时保留一个任意的 `task_index`）
  - 目的：**相同指令只调用一次 LLM，节省成本与时间**。

### 3.2 LLM 调用接口设计

- 抽象出一个统一接口，例如：

```python
def extract_objects_from_instruction(instruction: str) -> dict:
    ...
    return {
        "manipulated_object": str,
        "secondary_objects": List[str],
        "raw_response": str,
    }
```

- Prompt 设计要点（文本 LLM 即可，不强制多模态）：
  - 输入内容：
    - 原始英文/中英文任务指令（`task`）；
  - 要求模型：
    - 用 1 句简短自然语言说明“当前任务主要要操作的物体是什么”；
    - 用 JSON 形式输出：
      - `manipulated_object`: 一个短名词短语（如 `"green sponge"`, `"blue mug"`），**不得包含动词、地点、指令等**；
      - `secondary_objects`: 列出其他提及的关键物体名称（可为空列表）。
  - 强制输出结构化 JSON，便于解析。

### 3.3 失败与回退策略

- 若 LLM 输出解析失败、或 `manipulated_object` 为空 / `"unknown"`：
  - 该指令仍然写入 `task_object_prompts.jsonl`，但标记为不可用（如 `manipulated_object=""`）；
  - 后续阶段中，引用这个 `task_index` 的 episode 会被过滤掉或回退使用整句指令（可配置）。

### 3.4 输出格式

- 文件：`annotations/task_object_prompts.jsonl`
- 每行示例：

```json
{
  "task_index": 123,
  "task": "Pick up the green sponge and place it in the bowl.",
  "manipulated_object": "green sponge",
  "secondary_objects": ["bowl"],
  "llm_raw": "<原始完整响应，便于后续调试>"
}
```

---

## 4. 阶段二：Episode 级映射与过滤

### 4.1 构建 Episode ↔ Task ↔ Object 映射

- 读取：
  - `meta/episodes.jsonl`：每行包含 `episode_index`, `task_index` 等；
  - 阶段一的 `task_object_prompts.jsonl`。
- 对每个 `episode_index`：
  - 找到对应的 `task_index`；
  - 查映射表获得：
    - `instruction`（原始 task 文本）
    - `manipulated_object`（若为空视为失效）。

### 4.2 有效 episode 筛选

- 筛选条件（可配置，但建议默认）：
  1. `instruction` 非空；
  2. `manipulated_object` 非空；
  3. （可选）该 episode 出现在 `geo_segments.jsonl` 中，且有至少一个 segment；
  4. （可选）该 episode 出现在 steps PKL 中（和之前“有效 episode”定义保持一致）。

- 输出：
  - `annotations/episode_object_prompts.jsonl`：

```json
{
  "episode_index": 4567,
  "task_index": 123,
  "instruction": "Pick up the green sponge and place it in the bowl.",
  "manipulated_object": "green sponge"
}
```

> 后续 SAM3 标注阶段只遍历这份文件中的 episode，确保不会对“无指令 / 无可用物体名称”的 episode 浪费资源。

---

## 5. 阶段三：基于物体名称的 SAM3 BBox 标注

这一阶段整体沿用 `data_gen/bridge_sam3_bbox.py` 的设计，只在 **prompt 选择逻辑** 与 **输入源** 上做增强。

### 5.1 输入与整体流程

- 输入：
  - BRIDGE 数据根：`/share/project/baishuanghao/data/bridge_orig_lerobot`
  - 几何分段结果：`intermediate/geo_segments.jsonl`
  - Episode ↔ Object 映射：`annotations/episode_object_prompts.jsonl`
  - SAM3 模型权重：`--sam3_checkpoint`
- for 循环：
  1. 逐行读取 `geo_segments.jsonl` 中的 episode 记录；
  2. 查找该 `episode_index` 是否在 `episode_object_prompts.jsonl` 中：
     - 若不存在：跳过该 episode；
     - 若存在：获得 `manipulated_object`。
  3. 对 episode 内所有 `cycles[*].segments[*]`：
     - 选择关键帧 `frame_idx`（沿用 `select_keyframe`，默认为 segment 末帧）；
     - 从视频读取该帧图像；
     - 调用 SAM3 做 BBox 检测；
     - 将得到的 bbox 写入 `segment["grounding"]["bbox_2d"]` 等字段。
  4. 根据更新后的 segments、调用 `build_dense_active_bbox` 构建稠密 `dense_labels.active_bbox`。
  5. 将 enriched episode 写入输出 JSONL。

### 5.2 Prompt 选择策略

在现有 `choose_prompt` 基础上，改为优先使用 LLM 抽取的物体名：

```python
def choose_prompt_for_segment(segment, episode_prompt, default_prompt=None, instruction=None):
    if "target_object_ref" in segment and segment["target_object_ref"]:
        return segment["target_object_ref"]
    if episode_prompt:  # LLM 抽取的 manipulated_object
        return episode_prompt
    if default_prompt:
        return default_prompt
    return instruction or None
```

其中：

- `episode_prompt` = 来自 `episode_object_prompts.jsonl` 的 `manipulated_object`；
- `instruction` = 原始 task 文本（仅作为最后兜底）。

### 5.3 输出格式

- 输出文件：`annotations/segmentation_with_vlm_object_sam3.jsonl`
- 在原几何分段字段基础上新增：
  - 每个 segment：
    - `grounding`：
      - `bbox_2d`: `[ymin, xmin, ymax, xmax]` 或 SAM3 自带格式；
      - `confidence`: float；
      - `frame_idx`: int；
      - `prompt`: 实际使用的文本 prompt；
  - 顶层：
    - `dense_labels.active_bbox`: 长度为 `num_steps` 的 bbox 列表（或 `null`）；
    - `grounding_model`: `"sam3"`；
    - `object_prompt_source`: `"llm_instruction_extraction"`。

---

## 6. 组件划分与文件结构建议

为避免脚本过重、方便调试，建议拆成三个独立 CLI 脚本（可放在 `data_gen/`）：

1. `bridge_instruction_object_extraction.py`
   - 职责：从 `tasks.jsonl` 抽取物体名称，生成 `task_object_prompts.jsonl`。
2. `bridge_episode_object_mapping.py`
   - 职责：结合 `episodes.jsonl` 与阶段一结果，生成 `episode_object_prompts.jsonl`。
3. `bridge_sam3_bbox_from_objects.py`
   - 职责：基于 `geo_segments.jsonl` + episode-level object prompt 运行 SAM3，生成最终带 bbox 的 JSONL。

后续如需更细粒度（例如 segment 级 VLM 推理），也可以新增：

- `bridge_segment_vlm_reasoning.py`：
  - 对单段落图像 + 指令做推理，写入 `segment["target_object_ref"]` 和 `segment["segment_cot"]`；
  - 上述 `choose_prompt_for_segment` 已预留对 `segment["target_object_ref"]` 的优先级支持。

---

## 7. 调试与可视化计划

- 小规模 debug：
  - 首先在少量 episode 上跑完三个阶段；
  - 利用已有的 `visualize_segmentation.py` 增强版或新脚本：
    - 绘制 segment 边界 + SAM3 bbox；
    - 在图像上 overlay `manipulated_object` 文本，检查是否对齐任务语义。
- 质量检查：
  - 统计各类指标：
    - 有效 episode 数量；
    - 每个 episode 内有 bbox 的 segment 数量分布；
    - SAM3 置信度分布。
  - 抽样人工检查若干长尾 case（多目标、遮挡、模糊）。

---

## 8. 后续可能扩展

- Segment 级 VLM 推理：
  - 使用关键帧图像 + 指令，将“物体名称抽取”从 episode 级扩展到 segment 级；
  - 特别适合多物体交互场景（如先拿 sponge，再拿 cup）。
- 多相机联合：
  - 当前方案默认单机位（如 `observation.images.image_0`）；
  - 可以在 prompt 不变的情况下，对多机位帧都跑一遍 SAM3，选择 IoU 最大 / 置信度最高的结果。
- 时序一致性：
  - 在构建 `dense_labels.active_bbox` 时，对相邻帧做简单的线性插值 / 滑动平均，提升轨迹平滑度。

---

以上是基于现有 BRIDGE 数据与 `data_pipeline_bridge` 仓库的 **指令驱动 SAM3 BBox 标注**的完整方案草案。  
如果你觉得这个拆分与接口设计合理，下一步就可以按三个阶段依次实现对应的 Python 脚本。

