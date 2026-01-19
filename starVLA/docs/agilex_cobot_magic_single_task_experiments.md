# Agilex Cobot Magic：四个“单任务(单目录)”实验计划（最小改动）

## 目标
把当前 `agilex_cobot_magic_real4`（4 个目录混合训练）拆成 **4 个单目录实验**，分别只训练：
- `Agilex_Cobot_Magic_classify_object_fruit`
- `Agilex_Cobot_Magic_pour_water_twice`
- `Agilex_Cobot_Magic_stack_block_twice`
- `Agilex_Cobot_Magic_storage_object_two`

要求：尽可能复用现有训练入口（`train_ecot.py` + `scripts/run_starvla_agilex_cobot_magic.sh`），改动最小、最精炼。

---

## 现状约束（为什么不能“直接写目录名”）
数据加载入口在 `starVLA/dataloader/lerobot_datasets.py#get_vla_dataset`：
- 训练配置里只给一个 `datasets.vla_data.data_mix`
- 代码会直接索引 `DATASET_NAMED_MIXTURES[data_mix]`（定义在 `starVLA/dataloader/gr00t_lerobot/mixtures.py`）

因此：如果想“只跑某一个目录”，最小改动不是改 loader，而是 **在 mixtures 注册单目录的 mix key**，然后训练时把 `data_mix` 切换到对应 key。

---

## 最小改动方案（推荐）

### A. 仅改 1 个文件：新增 4 个单目录 `data_mix` key
修改文件：`starVLA/dataloader/gr00t_lerobot/mixtures.py`

在 `DATASET_NAMED_MIXTURES` 里新增 4 个 key（命名建议如下，清晰且短）：
- `agilex_cobot_magic_fruit`
- `agilex_cobot_magic_pour`
- `agilex_cobot_magic_stack`
- `agilex_cobot_magic_storage`

每个 key 的 value 都是单元素 list，robot_type 都保持为 `agilex_cobot_magic`，例如：
- `agilex_cobot_magic_fruit: [("Agilex_Cobot_Magic_classify_object_fruit", 1.0, "agilex_cobot_magic")]`

这样就无需改动 DataConfig / transforms / action horizon（50-step）等任何已有适配。

### B. 不改 YAML：用启动脚本的 `$@` 做 override（最省事）
你已经有 `scripts/run_starvla_agilex_cobot_magic.sh`，它会把 `$@` 透传给 `train_ecot.py`。

因此每个单任务实验只需要在启动命令里额外传：
- `--datasets.vla_data.data_mix <单任务key>`

### C. 强烈建议：每个单任务分开 steps cache 目录
原因：steps cache 虽然包含 `dataset_name`，但把 4 个实验混在同一个 cache 根目录里会让排查和复现实验更困难。

最小做法：每个实验在命令行 override 一个独立的目录：
- `--datasets.vla_data.bridge_annotations.steps_cache_path results/AgilexCobotMagic_final/steps_cache/agilex_cobot_magic_fruit`

### D. 强烈建议：每个单任务分开 `RUN_ID`
原因：日志、ckpt、wandb/run folder 一眼可区分。

最小做法：运行时设置环境变量（或直接在命令行 `--run_id`）：
- `RUN_ID=agilex_fruit_ecot`

---

## 具体执行清单（四个任务各一条命令）
共同前提：使用统一配置 `starVLA/config/training/agilex_cobot_magic_action_only_50step.yaml`，只通过 override 切换 `data_mix` 和 cache/run 命名。

下面假设你用默认脚本输出根目录：`RUN_ROOT_DIR=results/AgilexCobotMagic_final`。

1) Fruit
```bash
RUN_ID=agilex_fruit_ecot \
STEPS_CACHE_PATH=results/AgilexCobotMagic_final/steps_cache/agilex_cobot_magic_fruit \
bash scripts/run_starvla_agilex_cobot_magic.sh \
  --datasets.vla_data.data_mix agilex_cobot_magic_fruit
```

2) Pour
```bash
RUN_ID=agilex_pour_ecot \
STEPS_CACHE_PATH=results/AgilexCobotMagic_final/steps_cache/agilex_cobot_magic_pour \
bash scripts/run_starvla_agilex_cobot_magic.sh \
  --datasets.vla_data.data_mix agilex_cobot_magic_pour
```

3) Stack
```bash
RUN_ID=agilex_stack_ecot \
STEPS_CACHE_PATH=results/AgilexCobotMagic_final/steps_cache/agilex_cobot_magic_stack \
bash scripts/run_starvla_agilex_cobot_magic.sh \
  --datasets.vla_data.data_mix agilex_cobot_magic_stack
```

4) Storage
```bash
RUN_ID=agilex_storage_ecot \
STEPS_CACHE_PATH=results/AgilexCobotMagic_final/steps_cache/agilex_cobot_magic_storage \
bash scripts/run_starvla_agilex_cobot_magic.sh \
  --datasets.vla_data.data_mix agilex_cobot_magic_storage
```

备注：
- 如需分开端口：额外 `MASTER_PORT=2952X`。
- 如需固定同一个 ckpt 初始化：设置 `PRETRAINED_CKPT=...`（脚本已有该变量）。

---

## 可选：如果你要跑“四阶段 multistage”也拆成单任务
现状：`scripts/run_agilex_cobot_magic_multistage.sh` 没有透传 `$@`，所以不能像 final-stage 一样用命令行 override `data_mix`。

最小改动方向（二选一）：
1) **改 multistage 脚本 1 次**：新增一个 `DATA_MIX` 变量，并在 `TRAIN_CONFIG_ARGS`/cmd 中追加 `--datasets.vla_data.data_mix "${DATA_MIX}"`，然后四次运行分别设置 `DATA_MIX=...`。
2) **复制 4 份 multistage 脚本**：每份脚本里固定 `RUN_ID_PREFIX/STEPS_CACHE_DIR/DATA_MIX`（改动更分散，但每份更“傻瓜一键”）。

推荐 1)：改动集中且最小。

---

## 你跑之前我建议你确认的 3 件事（不会改代码）
1) 四个单任务的 `data_mix` key 在 `mixtures.py` 注册后，`train_ecot.py` 能正常构建 dataset（不再 KeyError）。
2) 每个实验的 `RUN_ID`/`STEPS_CACHE_PATH` 都不同（避免缓存/日志混淆）。
3) `delete_pause_frame` 仍保持 false（你的 Agilex action 是 vector 列；开启 pause filter 需要额外语义确认）。

