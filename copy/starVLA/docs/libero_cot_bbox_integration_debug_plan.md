# Libero LeRobot CoT/BBox 标注接入与验证规划

本文是一个**面向落地调试**的规划文档，用于验证：现有 StarVLA 的 LeRobot 数据加载管线是否能正确加载并使用你在 LIBERO 数据集上新增的 CoT（subtask/reasoning）与 BBox 标注（dense per-step bbox），并给出最小改动的配置策略与高效验证流程。

---

## 1. 背景与目标

### 1.1 背景

- 你已经在 LIBERO 的 LeRobot 数据集目录下标注了两类辅助数据：
  - CoT（subtask + reasoning + gripper_state）：例如 `annotations/episode_dense_captions_full.jsonl`
  - BBox（每 step 的 2D bbox）：例如 `annotations/episode_sam3_bboxes_from_dino.jsonl`
- 你希望复用现有训练框架（`datasets.vla_data.dataset_py: lerobot_datasets`）进行训练，重点验证**数据 load 与对齐**是否正确。

### 1.2 验证目标（可量化）

1. **加载正确**：dataloader 产出的 sample 中存在并正确填充：
   - `cot_available, cot_subtask, cot_reasoning, cot_gripper_state`
   - `bbox, bbox_valid, bbox_confidence`
2. **对齐正确**：同一 `(episode_index, step_index)` 的 CoT/BBox 与视频帧 step 对应正确，不存在系统性错位。
3. **覆盖率达标**：在你期望的训练子集上，`cot_available` 与 `bbox_valid` 的比例满足预期（例如多数 steps 为 True）。
4. **不污染其它数据集**：当 `data_mix` 包含多个 dataset 时，标注文件不会被错误复用到其它 dataset 上。

---

## 2. 当前 StarVLA 数据加载机制梳理（你需要知道的关键点）

### 2.1 走哪条管线？

- 只要配置了：
  - `datasets.vla_data.dataset_py: lerobot_datasets`
- 就会走 `starVLA/dataloader/lerobot_datasets.py:get_vla_dataset(...)`，内部创建 `LeRobotSingleDataset / LeRobotMixtureDataset`（位于 `starVLA/dataloader/gr00t_lerobot/datasets.py`）。

### 2.2 CoT/BBox 是怎么“挂到 sample 上”的？

- `LeRobotSingleDataset` 在初始化时会尝试加载 `BridgeAnnotations`：
  - 默认从 `<dataset_root>/annotations/...` 寻找文件
  - 或者通过 `datasets.vla_data.bridge_annotations.{cot_path,bbox_path}` 显式指定
- 在 `__getitem__` 里，会把标注写入 sample 字段：
  - `sample["cot_*"]` 与 `sample["bbox*"]`
- **重要**：名称叫 `BridgeAnnotations` 只是历史命名，逻辑上不限制只能用于 Bridge 数据集；只要文件存在且 schema 匹配，就能用于 Libero。

### 2.3 是否会把 CoT/BBox 注入 `language`？

- 由 `datasets.vla_data.bridge_reasoning.enable` 控制：
  - `enable: false`：sample 仍有 `cot_*`/`bbox*` 字段，但 `language` 不会被改写（模型可能“看不到”这些信息，除非后续模型输入构造显式用它们）。
  - `enable: true`：会使用 `BridgeReasoningFormatter` 把 bbox/subtask/reasoning 按 stage 规则编入训练文本（或转为 thinking token）。

---

## 3. 数据侧规范与约束（确保“能读 + 不错位”）

### 3.1 目录结构（LeRobot 标准）

以 `libero_10_no_noops_1.0.0_lerobot` 为例，至少需要：

- `<dataset_root>/meta/episodes.jsonl`
- `<dataset_root>/data/*/*.parquet`（动作/状态/语言等）
- `<dataset_root>/videos/...`（或其它视频后端可读的格式）
- `<dataset_root>/annotations/episode_dense_captions_full.jsonl`（你的 CoT）
- `<dataset_root>/annotations/episode_sam3_bboxes_from_dino.jsonl`（你的 BBox）

### 3.2 标注 schema 必须满足（与你当前标注一致）

**CoT JSONL（逐 episode 一行）**

- 顶层字段：`episode_index`, `steps`
- `steps` 是字典：`{ "<step_idx>": {"subtask": str, "reasoning": str, "gripper_state": int|None }, ... }`

**BBox JSONL（逐 episode 一行）**

- 顶层字段：`episode_index`, `num_steps`, `dense_labels.active_bbox`
- `dense_labels.active_bbox` 是长度为 `num_steps` 的 list：
  - 每个元素要么是 `null`，要么是 bbox 数组（期望 4 维 xyxy）

### 3.3 episode_index 对齐是硬约束

- dataloader 用的 key 是 `trajectory_id`（LeRobot 里通常就是 episode_index），并直接拿这个 int 去查 annotation。
- 所以你必须保证：
  - `annotations/*.jsonl` 的 `episode_index`
  - 与 `<dataset_root>/meta/episodes.jsonl` 的 `episode_index`
  - 完全一致。

### 3.4 bbox 坐标约定（需要你确认）

当前 loader 只是把 bbox 原样读出来塞进 sample（不会自动从像素坐标归一化）。

建议你明确并固定一种约定：
- **归一化 xyxy**（推荐）：`[x1, y1, x2, y2]` 在 `[0, 1]` 范围
- 或像素 xyxy：与原图分辨率相关（则后续使用时需要明确 resize/scale 规则）

验证阶段至少要检查：
- `bbox.shape == (4,)`
- `0 <= x1 < x2 <= 1`（若采用归一化）

---

## 4. 配置修改清单（最小集 + 推荐集 + 雷区）

### 4.1 最小必改（让 Libero 数据被读到）

在训练 YAML（或命令行 override）中，至少确保以下字段正确：

1. **启用 LeRobot dataloader**
   - `datasets.vla_data.dataset_py: lerobot_datasets`
2. **指向正确的数据根目录**
   - `datasets.vla_data.data_root_dir: /share/project/baishuanghao/data/libero_lerobot`
   - 注意：这里要指向“包含多个 dataset 文件夹的父目录”，而不是某个具体 dataset 目录
3. **选择合适的数据 mix**
   - 若只先验证 libero10：`datasets.vla_data.data_mix: libero_10`
   - 若想混合训练：`libero_all`（见 4.3 雷区）
4. **显式指定 annotation 文件路径（强烈推荐）**
   - `datasets.vla_data.bridge_annotations.cot_path: annotations/episode_dense_captions_full.jsonl`
   - `datasets.vla_data.bridge_annotations.bbox_path: annotations/episode_sam3_bboxes_from_dino.jsonl`

> 为什么推荐“相对路径”？因为它会以每个 dataset 的 `dataset_root` 为基准解析，天然避免多数据集混合时互相串标注的问题。

### 4.2 调试推荐配置（让验证更省事）

**A. 先宽松不过滤（验证“能否加载”）**

- `datasets.vla_data.bridge_annotations.filters.require_cot_episode: false`
- `datasets.vla_data.bridge_annotations.filters.require_bbox_episode: false`
- `datasets.vla_data.bridge_annotations.filters.require_bbox_step: false`
- `datasets.vla_data.bridge_annotations.filters.min_episode_bbox_coverage: null`

**B. 再逐步收紧（验证“覆盖率/对齐”）**

- episode 级过滤：
  - `require_bbox_episode: true`
  - 或 `min_episode_bbox_coverage: 0.5`（示例）
- step 级过滤（最严格）：
  - `require_bbox_step: true`

**C. 将 CoT/BBox 注入 language（让肉眼检查更直观）**

- `datasets.vla_data.bridge_reasoning.enable: true`
- `datasets.vla_data.bridge_reasoning.stage: 1`

`stage: 1` 的优势：文本里显式出现 `BBox/Subtask/Reasoning`，最利于对齐排查。等确认无误后再切换到训练真正需要的 latent stage。

### 4.3 高危雷区（最常见的“看似能跑、实际错了”）

1. **在 multi-dataset 的 `data_mix` 下使用绝对路径的 `cot_path/bbox_path`**
   - 风险：其它 dataset 也会去加载同一个绝对路径文件（存在即加载），造成标注“串台”。
   - 解决：用相对路径，或确保每个 dataset 都有对应文件并且路径相对 dataset_root。
2. **把 `data_root_dir` 指到 dataset 目录而不是父目录**
   - 风险：mixture 里拼路径会变成 `<dataset_root>/<dataset_name>`，导致找不到数据。
3. **默认文件名不匹配但忘了 override**
   - 风险：pipeline 会静默地“没加载标注”，训练照跑但完全没用到 CoT/BBox。
4. **`delete_pause_frame` 导致你以为 bbox 覆盖低**
   - 如果删除 pause frame 后采样 step 子集变化，bbox_valid 覆盖率统计会变（但不一定是标注错）。
   - 建议：验证时先固定 `delete_pause_frame`（比如先关掉）再对比。

---

## 5. 高效验证流程（从 5 分钟到 1 小时的分层检查）

### 5.1 第 0 层：静态文件检查（最快）

目的：排除“路径/文件不存在/schema 不对”的低级错误。

检查项：
- 两个 annotation 文件存在且非空
- CoT 文件首行能解析 JSON，并包含：`episode_index`, `steps`
- BBox 文件首行包含：`episode_index`, `num_steps`, `dense_labels.active_bbox`
- `dense_labels.active_bbox` 的长度是否等于 `num_steps`

产出物（记录到 log/笔记里）：
- `episode_index=0` 的 `num_steps`
- 首个非空 bbox 的示例值

### 5.2 第 1 层：单样本 load 检查（最关键、信息密度最高）

目的：确认 dataloader 确实把标注挂到了 sample，并且字段符合预期。

做法：
- 用你的训练配置（或等价的 override）构建 dataset
- 直接取 `dataset[0]`（或随机 index）并打印/断言关键字段：
  - `cot_available` 是否为 True
  - `cot_subtask/cot_reasoning` 是否非空（至少有一部分 steps 非空）
  - `bbox_valid` 是否为 True
  - `bbox` 是否为 4 维、数值范围是否合理
  - 若开启 `bridge_reasoning.enable`：`language` 是否出现 BBox/Subtask/Reasoning（stage=1 时）

建议额外检查：
- 同一个 `trajectory_id` 的几个相邻 step（例如 base_index=0/1/2）是否有“连续合理变化”，排除系统性错位。

### 5.3 第 2 层：小批量统计（验证覆盖率 + 发现系统性异常）

目的：用统计数据定位“只少数 step 有 bbox”“绝大多数 bbox 都是 0”“cot 全空”等系统性问题。

建议统计（例如扫前 200~2000 个 samples）：
- `cot_available` 比例
- `bbox_valid` 比例
- `bbox` 的 min/max（分别对 x1,y1,x2,y2）
- `bbox` 是否频繁出现全 0（通常表示没读到或被判 invalid）

### 5.4 第 3 层：端到端 smoke test（可选）

目的：确认训练前向不会因为文本长度/特殊 token/bbox dtype 等细节崩溃。

策略：
- 不跑完整训练，只跑：
  - dataloader -> 取 1 个 batch -> model forward（不反传）
- 这一步通常不是主要矛盾，但能尽早发现“数据没错、输入拼接阶段出错”的情况。

---

## 6. 问题定位决策树（快速缩小范围）

### 6.1 `cot_available` 几乎全 False

优先排查：
1. `bridge_annotations.cot_path` 是否正确（尤其相对路径是否相对 dataset_root）
2. CoT JSONL 的 `episode_index` 是否与 `meta/episodes.jsonl` 对齐
3. `steps` 的 key 是否是可转 int 的字符串（例如 `"0"`, `"1"`）

### 6.2 `bbox_valid` 几乎全 False 或 bbox 全 0

优先排查：
1. `bridge_annotations.bbox_path` 是否正确
2. `dense_labels.active_bbox` 是否存在且为 list
3. list 里 bbox 是否为 4 元素且可转 float
4. 如果你打开了 step/episode 过滤：是否把数据过滤光了（看初始化时 all_steps 数量）

### 6.3 开启 `bridge_reasoning.enable` 后 `language` 没变化

优先排查：
1. `bridge_reasoning.enable` 是否真的为 true
2. `stage` 是否为 0/1（stage>=2 可能变成 thinking token，不易肉眼识别）
3. sample 的 `language` 在后处理里是否被其它逻辑覆盖（例如上游/下游再改写）

### 6.4 混合数据集时结果异常（强烈怀疑串标注）

典型症状：
- 其它 libero 子集（object/goal/spatial）也“莫名其妙”出现 libero10 的标注风格/内容

排查与修复：
- 检查 `cot_path/bbox_path` 是否写了绝对路径
- 改为相对路径并确保各 dataset 自己的 `annotations/` 下都放有对应文件（或只训练你确实标注过的子集）

---

## 7. 交付物与里程碑（建议）

### 7.1 里程碑 M1：能加载（半天内）

- 单样本检查通过
- 小批量统计中 `cot_available`/`bbox_valid` 符合预期范围

### 7.2 里程碑 M2：能稳定训练（1 天内）

- 端到端 smoke test 通过
- 正式训练启动后前若干 step loss 正常（无 NaN/无异常爆炸）

### 7.3 建议保留的调试产出

- 一份小批量统计结果（文本或 json）
- 若采用 stage=1：保存若干条 `language` 文本示例用于人工 spot-check

---

## 8. 附录：最小 override 清单（模板）

> 下面是“你需要覆盖哪些配置字段”的模板（用于 YAML 或命令行 override）。请按你的实际路径替换。

**数据集根与 mix**

- `datasets.vla_data.dataset_py = lerobot_datasets`
- `datasets.vla_data.data_root_dir = /share/project/baishuanghao/data/libero_lerobot`
- `datasets.vla_data.data_mix = libero_10`

**标注文件（推荐相对路径）**

- `datasets.vla_data.bridge_annotations.cot_path = annotations/episode_dense_captions_full.jsonl`
- `datasets.vla_data.bridge_annotations.bbox_path = annotations/episode_sam3_bboxes_from_dino.jsonl`
- （可选）`datasets.vla_data.bridge_annotations.steps_cache_path = /your/writable/path/steps_libero_goal.pkl`
- （可选）`datasets.vla_data.bridge_annotations.write_steps_cache = true`（默认 false；开启后允许把计算出来的 steps 缓存写到上面的自定义路径）

**调试期 filters（先宽松）**

- `datasets.vla_data.bridge_annotations.filters.require_cot_episode = false`
- `datasets.vla_data.bridge_annotations.filters.require_bbox_episode = false`
- `datasets.vla_data.bridge_annotations.filters.require_bbox_step = false`
- `datasets.vla_data.bridge_annotations.filters.min_episode_bbox_coverage = null`

**肉眼可读的文本注入（可选但强烈推荐用于调试）**

- `datasets.vla_data.bridge_reasoning.enable = true`
- `datasets.vla_data.bridge_reasoning.stage = 1`
- `datasets.vla_data.bridge_reasoning.include_bbox = true`

---

## 9. 本仓库的最小落地实现（已新增）

- Debug 配置：`starVLA/config/training/libero_goal_cot_bbox_debug.yaml`
- Debug 脚本：`scripts/debug/check_libero_goal_dataloader.py`
- 运行示例：`python scripts/debug/check_libero_goal_dataloader.py --num_samples 200 --print_examples 5`
  - 注意：不同 LIBERO 子集的 bbox 文件名可能不同（例如 `episode_sam3_bboxes_from_dino_final.jsonl`），以各自 `annotations/` 目录实际文件为准。
