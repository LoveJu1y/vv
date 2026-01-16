# Agilex Cobot Magic（LeRobot）50-step Action 训练适配计划（仅规划，先不改代码）

## 0. 目标与约束（本计划的“定论”）

### 数据集选择
- 数据根目录：`/share/project/baishuanghao/data/BAAI_real_world_lvjing_lerobot`
- **只使用名字不包含 `v3.0` 的 4 个数据集**：
  - `Agilex_Cobot_Magic_classify_object_fruit`
  - `Agilex_Cobot_Magic_pour_water_twice`
  - `Agilex_Cobot_Magic_stack_block_twice`
  - `Agilex_Cobot_Magic_storage_object_two`

### Action 监督窗口（最关键的对齐定义）
- 使用 “**当前 + 未来49**” 的 50 步监督：`action_indices = list(range(50))`
- 模型侧严格对应：
  - `future_action_window_size = 49`
  - `action_horizon = 50`

### Gripper（二值化）
- **左右手 gripper 都要二值化**（action 里两维）
- 使用 raw 值阈值 `0.05`（不是归一化后的值）
- 之前是 `0.5`，这次要改为 `0.05`

### 训练窗口/采样策略
- “libero 和 bridge 怎么做就怎么做”：保持现有 LeRobot realdata 的窗口/补齐逻辑，不引入新的丢尾部策略。
- 当前 LeRobot realdata 对 state/action 越界步的 padding 是 **zero padding**（见 `starVLA/dataloader/gr00t_lerobot/datasets.py:1351`）。

### State 是否使用
- **不使用 state 输入**（因此 state 的二值化/归一化对训练不敏感，但 action 的二值化必须正确）。

---

## 1. 数据集元信息需要核对什么（确保 key/切片/语言都能对上）

每个数据集目录都包含 `meta/` 与 `data/*/*.parquet`，需要重点核对：

### 1.1 `meta/modality.json`（决定 key 如何映射到 parquet 字段与切片）
必须确认它们四个数据集一致，并且满足：
- `video`：
  - `video.ego_view` → `observation.images.center_camera`
  - `video.left_view` → `observation.images.left_camera`
  - `video.right_view` → `observation.images.right_camera`
- `state`：来自 `observation.state` 的切片（非连续索引也允许）
- `action`：来自 `action` 的切片（总维度 14）
  - left_arm: [0:6], left_hand: [6:7], right_arm: [7:13], right_hand: [13:14]
- `annotation`：
  - `annotation.human.action.task_description` 的 `original_key` 应为 `task_index`

### 1.2 `meta/info.json`（决定视频读取与 dataset schema）
需要确认：
- fps、视频分辨率、image keys 是否一致（避免同一个 DataConfig 无法覆盖全部）
- `features.action` shape 为 `[14]`（与上面的 action 切片对齐）
- `features.observation.state` shape 为 `[128]`

### 1.3 语言映射链路（非常容易“看起来有 language 其实是数字”）
当前管线是：
- parquet 里读 `task_index`
- `meta/tasks.jsonl` 里用 `task_index -> task(string)` 映射
- 最终 `annotation.human.action.task_description` 返回的是 task 字符串列表
对应代码路径：
- tasks 加载：`starVLA/dataloader/gr00t_lerobot/datasets.py:895`
- language 读取并映射：`starVLA/dataloader/gr00t_lerobot/datasets.py:1355`

---

## 2. DataConfig（你给的版本）作为实现目标的可行性结论

### 2.1 key 设计
你的 key 设计在当前 LeRobot 管线下是合理的（会在初始化时用 `meta/modality.json` 做 integrity check）：

```python
video_keys = ["video.ego_view", "video.left_view", "video.right_view"]
state_keys = ["state.left_arm", "state.left_hand", "state.right_arm", "state.right_hand"]
action_keys = ["action.left_arm", "action.left_hand", "action.right_arm", "action.right_hand"]
language_keys = ["annotation.human.action.task_description"]
observation_indices = [0]
action_indices = list(range(50))  # 当前 + 未来49
```

### 2.2 维度对齐（必须写进训练 YAML）
- action concat 后维度：6 + 1 + 6 + 1 = **14**
- 若 state 也 concat，同样是 14（但我们不使用 state 输入）
- 因此训练配置必须设置：
  - `framework.action_model.action_dim = 14`
  - `framework.action_model.state_dim = 14`（即便不喂 state，也建议保持一致，减少未来切换风险）

---

## 3. 50-step 对齐在代码/配置里涉及哪些“必须一起改”的点

### 3.1 数据侧（DataConfig）
- `action_indices = range(50)` 决定 dataloader 返回的 action 序列长度是 50

### 3.2 模型侧（ActionHead/QwenGR00T）
目前 `QwenGR00T` 用配置里的 `future_action_window_size` 截取监督标签：
- label 截取：`starVLA/model/framework/QwenGR00T.py:238`
- chunk_len 计算：`starVLA/model/framework/QwenGR00T.py:71`
因此要 50-step 必须对齐：
- `future_action_window_size = 49`
- `action_horizon = 50`

### 3.3 ActionHead 推理侧也依赖 `action_horizon`
`FlowmatchingActionHead.predict_action()` 里直接用：
- `actions = torch.randn((B, action_horizon, action_dim))`（见 `starVLA/model/modules/action_model/GR00T_ActionHeader.py:627`）
所以 `action_horizon=50` 也会影响推理时的采样长度。

---

## 4. Gripper 二值化阈值 0.05：需要怎么做才不破坏其他数据集

### 4.1 现状：binary 阈值硬编码为 0.5（需要调整）
当前二值化逻辑写死在 transform 里：
- `StateActionNormalization(mode="binary")`：`(x > 0.5)`（`starVLA/dataloader/gr00t_lerobot/transform/state_action.py:185`）
- `inverse()` 同样 `x > 0.5`（`starVLA/dataloader/gr00t_lerobot/transform/state_action.py:209`）

### 4.2 推荐方案（优先）：让阈值可配置（默认 0.5，不影响老实验）
为了不影响 libero/bridge 等已有训练结果，建议实现为：
- binary 模式支持可配置阈值（默认 `0.5`）
- 并允许按 key 指定阈值（例如只对 `action.left_hand/right_hand` 使用 `0.05`）

落地点（后续实现时选一种）：
1) 在 `StateActionTransform` 里新增参数：`binary_thresholds: dict[str, float]`  
2) 或在 normalization mode 里扩展为 `binary@0.05` 这种语法（解析后传给 normalization）

本次需求：仅对新数据集的 hand action keys 使用 `0.05`：
- `action.left_hand`
- `action.right_hand`

### 4.3 语义一致性检查（必须保持与旧阈值方案同一方向）
需要确认二值化后的 0/1 含义与控制端一致（例如 1 是 close 还是 open），并与之前 0.5 阈值方案保持同一语义方向。

---

## 5. 需要新增的训练配置与 dataloader 接入点（实现清单，后续你确认后再动手）

### 5.1 新 robot_type（否则 dataloader 无法选择你的 DataConfig）
入口：
- `starVLA/dataloader/lerobot_datasets.py:32`：`ROBOT_TYPE_CONFIG_MAP[robot_type]`
计划：
- 在 `starVLA/dataloader/gr00t_lerobot/data_config.py` 新增 `AgilexCobotMagicDataConfig`
- 加入 `ROBOT_TYPE_CONFIG_MAP`（`starVLA/dataloader/gr00t_lerobot/data_config.py:597`）
- （可选）在 `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py:67` 增加映射，避免 warning

### 5.2 新 dataset mix（让 YAML 一行引用 4 个数据集）
入口：
- `DATASET_NAMED_MIXTURES`：`starVLA/dataloader/gr00t_lerobot/mixtures.py:13`
计划：
- 新增一个 mix（例如 `agilex_cobot_magic_real4`）
- 包含四项 `(dataset_name, weight, robot_type="agilex_cobot_magic")`

### 5.3 新训练 YAML（仿照 libero/bridge 的 horizon 对齐写法）
参考：
- libero 8：`starVLA/config/training/libero_all_ecot_stage4.yaml:99`
- bridge 16：`starVLA/config/training/bridge_lerobot_stage2.yaml:87`
计划新增 YAML（命名待定，例如）：
- `starVLA/config/training/agilex_cobot_magic_ecot_stageX.yaml`
其中必须包含：
- `data.data_root_dir = /share/project/baishuanghao/data/BAAI_real_world_lvjing_lerobot`
- `data.data_mix = agilex_cobot_magic_real4`
- `framework.action_model.action_dim = 14`
- `framework.action_model.state_dim = 14`
- `framework.action_model.future_action_window_size = 49`
- `framework.action_model.action_horizon = 50`
- 结合显存/吞吐，调整 batch size 与 grad accumulation（50-step 显著更吃显存）

---

## 6. 验证（在大训练前，必须做的“链路正确性”检查）

### 6.1 Dataloader 冒烟（shape/dtype/内容）
目标：确认 batch 中各模态 shape 正确，且语言是字符串：
- video：3 路
- action：长度 50、维度 14
- language：能读到 task 字符串而不是数字

### 6.2 Gripper 二值化检查（阈值 0.05 是否过激）
对四个数据集各抽样统计：
- `action.left_hand/right_hand` raw 值分布与 (x > 0.05) 后的 0/1 比例
若出现几乎全 0 或全 1，需要重新确认阈值/语义（否则训练会退化）。

### 6.3 小规模 sanity train
先用单个数据集、极少 steps：
- 确认不报 shape 错误
- 确认 loss 能下降（至少不发散/不恒定）
再扩到 4 数据集 mix。

---

## 7. 实施顺序（你确认后我按这个顺序改，改每一步前会先向你示意）

1) 新增 `AgilexCobotMagicDataConfig` 与 `ROBOT_TYPE_CONFIG_MAP` 接入（不改现有默认行为）
2) 新增 dataset mix：`agilex_cobot_magic_real4`
3) binary 阈值“可配置化”，并在新 DataConfig 对 hand keys 指定 `0.05`
4) 新训练 YAML：50-step/horizon/dim 对齐写死
5) 冒烟验证与小训练验证

> 说明：本文件是“计划文档”。你确认后我才开始动代码，并且每次要改哪个文件、改什么，会先在对话里明确告知。

---

## 8. 具体代码修改 Schema（最小侵入、默认行为不变）

本节把“将要改哪些文件、改哪些类/字段、预期输入输出”写清楚，便于你 review。**除新数据集适配外，不改现有 libero/bridge 行为**。

### 8.1 `starVLA/dataloader/gr00t_lerobot/transform/state_action.py`

#### 8.1.1 目标
- 让 `binary` 归一化的阈值从“写死 0.5”变为“默认 0.5，但可对指定 key 配置成 0.05”。
- 若不配置阈值，**行为与现在完全一致**。

#### 8.1.2 修改点（类/字段级别）
1) `class Normalizer`
   - 现在：`binary` 模式在 `forward()` / `inverse()` 中硬编码 `(x > 0.5)`。
   - 计划：新增一个参数（默认值不变）：
     - `binary_threshold: float = 0.5`
   - 行为：
     - `forward()`：`(x > self.binary_threshold)`
     - `inverse()`：`(x > self.binary_threshold)`（保持现有“binary inverse 也是阈值化”的风格）

2) `class StateActionTransform`
   - 计划新增一个可选字段（默认空字典）：
     - `binary_thresholds: dict[str, float] = {}`
   - 用法：当 `normalization_modes[key] == "binary"` 时：
     - 取 `binary_thresholds.get(key, 0.5)` 作为该 key 的阈值，传给对应的 `Normalizer`。
   - 兼容性：
     - 老配置不传 `binary_thresholds`，则所有 binary 仍用 0.5（不影响 libero/bridge 旧实验）。

> 注：这里不引入新的“mode 字符串语法”（如 `binary@0.05`），而是用单独字段，改动更小、可读性更高、也更不容易破坏旧配置。

---

### 8.2 `starVLA/dataloader/gr00t_lerobot/data_config.py`

#### 8.2.1 目标
- 新增一个 `AgilexCobotMagicDataConfig`（覆盖 4 个 non-v3.0 数据集）。
- 注册为新的 `robot_type="agilex_cobot_magic"`，让 `starVLA/dataloader/lerobot_datasets.py` 能通过 `ROBOT_TYPE_CONFIG_MAP[robot_type]` 找到它。

#### 8.2.2 新增类（骨架与关键参数）
新增 `class AgilexCobotMagicDataConfig`，核心字段固定为：
- `video_keys = ["video.ego_view", "video.left_view", "video.right_view"]`
- `state_keys = ["state.left_arm", "state.left_hand", "state.right_arm", "state.right_hand"]`（虽然我们不使用 state 输入，但保留一致性）
- `action_keys = ["action.left_arm", "action.left_hand", "action.right_arm", "action.right_hand"]`
- `language_keys = ["annotation.human.action.task_description"]`
- `observation_indices = [0]`
- `action_indices = list(range(50))`（当前 + 未来49）

`transform()` 的关键点：
- arms：`q99`
- hands：`binary`
- 并且对 `action.left_hand` / `action.right_hand` 指定阈值 `0.05`（通过 8.1 引入的 `binary_thresholds`）

#### 8.2.3 注册 robot_type
在 `ROBOT_TYPE_CONFIG_MAP` 增加一项：
- key：`"agilex_cobot_magic"`
- value：`AgilexCobotMagicDataConfig()`（或保持与文件里现有风格一致的实例写法）

---

### 8.3 `starVLA/dataloader/gr00t_lerobot/mixtures.py`

#### 8.3.1 目标
新增一个 “4 个 non-v3.0 数据集” 的 mix，给训练 YAML 一行引用。

#### 8.3.2 修改点
在 `DATASET_NAMED_MIXTURES` 增加一个条目（命名可最终确认）：
- `agilex_cobot_magic_real4 = [`
  - `("Agilex_Cobot_Magic_classify_object_fruit", 1.0, "agilex_cobot_magic")`
  - `("Agilex_Cobot_Magic_pour_water_twice", 1.0, "agilex_cobot_magic")`
  - `("Agilex_Cobot_Magic_stack_block_twice", 1.0, "agilex_cobot_magic")`
  - `("Agilex_Cobot_Magic_storage_object_two", 1.0, "agilex_cobot_magic")`
  - `]`

兼容性：只新增，不改旧 mix。

---

### 8.4 `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py`（可选，但建议）

#### 8.4.1 目标
避免 dataloader 打 warning，并给新 robot_type 一个明确的 embodiment tag。

#### 8.4.2 修改点
在 `ROBOT_TYPE_TO_EMBODIMENT_TAG` 增加：
- `"agilex_cobot_magic": EmbodimentTag.NEW_EMBODIMENT`（或你希望的 tag）

> 这一步是“锦上添花”，即便不做也能跑，只是会打印 warning。

---

### 8.5 新增训练 YAML（仿照 libero/bridge 对齐规则）

#### 8.5.1 文件新增
新增：`starVLA/config/training/agilex_cobot_magic_ecot_stageX.yaml`（`stageX` 你决定用哪套 stage 体系）

#### 8.5.2 必须写死的对齐字段
数据：
- `data.data_root_dir: /share/project/baishuanghao/data/BAAI_real_world_lvjing_lerobot`
- `data.data_mix: agilex_cobot_magic_real4`

Action head：
- `framework.action_model.action_dim: 14`
- `framework.action_model.state_dim: 14`
- `framework.action_model.future_action_window_size: 49`
- `framework.action_model.action_horizon: 50`

#### 8.5.3 其他字段处理原则
- 从现有模板（如 `starVLA/config/training/libero_all_ecot_stage4.yaml`）复制：只改 data_root_dir / data_mix / action_dim/state_dim/horizon，其他尽量不动。
- batch size、grad accumulation：只做“为了跑得动 50-step”所需的最小调整（不做无关重构）。

