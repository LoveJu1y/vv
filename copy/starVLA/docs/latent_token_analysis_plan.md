# Latent Reasoning Token Analysis Plan (StarVLA / Libero)

## 0) 背景与目标（你要回答的两个问题）

你当前的“隐式推理”实现是：在 prompt 中插入 `<|start_of_thinking|> <|thinking|>... <|end_of_thinking|>`，并在 `QWen3.forward_latent()` 中用 KV-cache 做多次短 forward，把“前一位置的 hidden state”回填到 `<|thinking|>` 的 `inputs_embeds`，得到最终的 `hidden_states` 供 action head cross-attn 使用。

本分析实验聚焦两个核心问题：

1. **表征塌缩（Representation Collapse）**  
   你的 3 个 latent token 是否学到“彼此不同且随样本变化”的表示？若三者几乎恒等/低秩，则说明 latent 可能没有承担隐式 CoT 的功能（或者只提供 compute scale）。

2. **任务/能力分工（Role Specialization）**  
   在不同任务（尤其 Libero 不同 suite / task）下，这些 latent token 的 hidden state 是否呈现系统性差异，且不同 token 对不同因素（子任务/目标位置/长程规划）有不同侧重？

> 注意：在当前实现中，3 个 latent 并不是 3 个不同 token id，而是 **同一个 `<|thinking|>` token id 的重复**；分工主要靠“位置/上下文”实现。

---

## 1) 选用数据源（先用 Libero 训练数据）

**建议：优先用 Libero 的训练数据（LeRobot 格式）作为第一轮 analysis**，原因：
- 任务集合离散且有 suite/task_id/task_description，适合做“按任务分组”的分布统计。
- 更容易做“受控采样”（同一 task 多 episode/step），把差异归因到任务因素而非数据混杂。

后续第二轮再用 Bridge（更杂、跨 embodiment）验证泛化即可。

---

## 2) 实验总体分阶段（从最简单到更强结论）

## 2.1) 落地方式（完全复用 `train_ecot.py` 管线）

我们将**不依赖环境评测脚本**来做 latent 分析，而是复用训练管线的三件套：
`build_dataloader(...)` → `QwenGR00T.forward(...)` → loss/优化器（可选）。

核心落点：
- 在 `starVLA/model/framework/QwenGR00T.py` 的 `forward()` 中，`forward_latent()` 返回后已经有：
  - `qwen_inputs["input_ids"]`：可定位 `<|thinking|>` / `<img_next>` 的位置；
  - `last_hidden`：可直接抽取这些位置的 hidden state 做统计。
- 在 `starVLA/training/train_ecot.py` 的 `_train_step()` 中，给 `self.model.forward(...)` 传入 `global_step`/rank 信息，
  让分析逻辑做到“每 K step 触发一次，且只由 main process 落盘”，不干扰训练吞吐。

为兼顾“先验证机制”和“批量统计”，建议拆成三种运行模式：
1. **Debug 模式（一次 batch）**：用 VSCode 断点确认 token 对齐与 `forward_latent` 逻辑正确。
2. **Analysis-only 模式（无反传）**：只跑 dataloader 前 N 个 batch，`model.eval() + torch.no_grad()`，输出统计文件。
3. **Training + 周期性分析**：正常训练，每隔 K step 采样 1 个 batch 计算统计并落盘，得到随训练演化曲线。

> 说明：当前 `QwenGR00T._extract_reasoning_mask()` 默认只在 `use_reasoning_summary/use_reasoning_film` 打开时返回。
> 因此分析时不要依赖 `reasoning_mask`，而应直接用 `qwen_inputs["input_ids"] == thinking_token_id` 来定位 thinking tokens。

---

### Phase A — Sanity Check：确认 latent 结构真的生效

目标：在不跑完整评测的情况下，确认“token 对齐 + forward_latent 路径 + hidden 可提取”都正确。

步骤（建议固定抽样规模：例如 2 个 suite × 每个 suite 2 个 task × 每个 task 20 个 step）：

1. **样本字段完整性**
   - 检查 sample 中是否包含：`lang/language`（已格式化）、`image`、（可选）`bbox/cot_subtask/cot_reasoning/action_tokens`。
   - 确认 `language` 中存在 `". @ "` delimiter（训练 formatter 的约定）。

2. **Token 计数与定位**
   - 用 `qwen_vl_interface.build_qwenvl_inputs(...)` 得到 `input_ids`。
   - 统计并记录每个 sample：
     - `<|thinking|>` 数量
     - `<|start_of_thinking|>` 与 `<|end_of_thinking|>` 是否存在
     - `<img_next>` 数量（若启用）
   - 期望：
     - 若采用 stage4 且 subtask/reason 非空：通常会出现 3 个 `<|thinking|>`（受 `tag2think_count` 与字段是否为空影响）。
     - `<img_next>` 通常为 16（由配置决定）。

3. **确保走 `forward_latent`**
   - 在同一 batch 上分别运行：
     - 单次 forward（baseline）
     - `forward_latent`（implicit）
   - 记录：
     - `num_reasoning_passes` 是否为 `n_thinking + 1`
     - 逻辑一致性：thinking token 数为 3 时，passes 应为 4。

4. **最粗粒度表征非退化检查**
   - 对同一样本的 3 个 thinking token hidden state，算 `cos(t1,t2)`, `cos(t2,t3)`, `cos(t1,t3)`。
   - 预期：不应全部接近 1.0（完全相同）；若接近 1.0，优先排查 token 对齐/切片错误或 latent 更新没生效。

输出物（建议落盘到同一目录，方便复现对照）：
- `samples_meta.jsonl`：每条样本的 suite/task/episode/step、token 计数、是否包含 cot 字段等。
- `sanity_token_counts.csv`：聚合统计（均值/方差/分位数）。

---

### Phase B — Collapse Test：判断是否“只在 scale 上有效”

目标：用定量指标回答“latent 是否塌缩”，而不是只看 PCA 图。

指标建议（按 token 位置分别统计）：

1. **Token 内塌缩（within-sample collapse）**
   - 同一样本中，三两两相似度分布（cosine / L2）。
   - 可按 suite/task 分组作箱线图。

2. **Token 间塌缩（across-sample collapse）**
   - 对每个 token 位置（第 1/2/3 个 thinking token），计算跨样本：
     - 均值向量范数、逐维方差
     - covariance 的有效秩（effective rank）或 PCA explained variance（如前 1/2 主成分占比）
   - 对照组：取 thinking token 邻近的普通文本 token hidden state 做同样统计，避免误判“整句都低方差”。

3. **训练步/ckpt 维度（可选但很强）**
   - 同一套指标在多个 ckpt 上画趋势：如果越训越塌缩，说明 latent 容易退化成“占位/计算”。

输出物：
- `collapse_metrics_{suite}.csv`
- `collapse_plots/`（相似度分布图、effective-rank 趋势图）

---

### Phase C — Role Specialization：任务差异与分工证据

目标：证明“不同任务下 latent 表示不同”，且不同 token 的差异模式不同。

#### 采样策略（避免混杂）
优先做“受控变化”：
- **跨 suite**：`libero_spatial vs libero_goal vs libero_object`（suite 级差异大）
- **同 suite 不同 task**：固定 suite，挑 3–5 个 task
- **同 task 多 episode/step**：把随机性平均掉，观察稳定差异

#### 分析方法（从易到难）
1. **可视化（展示用）**
   - PCA/UMAP：分别对 token1/token2/token3 的 embedding 做降维，颜色标注 suite/task。
   - 重点：看“token 间分工”——token1/2/3 是否呈现不同的分簇方式。

2. **定量可分性（更可信）**
   - 线性 probe（建议先做最简单的）：
     - 输入：单个 token hidden（t1/t2/t3）或三者拼接
     - 预测：suite id、task id（或 task_description hash）
   - 指标：accuracy / macro-F1（分类）或 NMI / silhouette（聚类）

3. **按“预期因素”对齐（可选）**
   - 若你希望 token2 更偏空间/目标：可用 bbox_valid、bbox 的粗分桶（左/中/右，上/中/下）做弱监督 probe。
   - 若你希望 token1 更偏 subtask：用任务内部阶段标签（若有）或用文本模板/动作 token 分桶做代理标签。

输出物：
- `embedding_dump.pt`（或 npy）：保存 token embedding 与 meta（suite/task/episode/step）
- `pca_umap_plots/`
- `probe_results.csv`

---

### Phase D — 因果/消融（可选，但能一锤定音）

目标：回答“latent 变了 ≠ latent 被用上”。通过干预 latent hidden state，观察策略性能或 action 输出变化。

建议只做最小版（先离线看 action 输出差异，再上环境成功率）：
1. **Intervention types**
   - `zero`: 将 3 个 thinking token 的 hidden state 置零/置均值
   - `swap`: 交换 token 顺序（t1↔t3 等）
   - `shuffle`: 跨样本打乱 thinking token（破坏与当前样本绑定）
2. **观测量**
   - 不跑环境时：看 action head 输出（normalized_actions）的分布变化、与原始输出的 L2 差。
   - 跑少量环境时：每 suite 选 2 个 task、每 task 10 rollouts，比较 success drop。

输出物：
- `intervention_action_delta.csv`
- （可选）`intervention_success_{suite}.json`

---

## 3) 关键实现细节与注意事项（避免踩坑）

1. **3 个 latent token 数量可能不恒定**
   - 训练数据 stage4 时，`BBOX` 一般总会被加入（无 bbox 时用 full-image bbox），但 `SUBTASK/REASON` 可能为空导致缺失。
   - 若你的分析强依赖“固定 3 token”，建议先过滤样本：`cot_subtask` 与 `cot_reasoning` 非空。

2. **token 对齐依赖 build 输入路径**
   - `QWen3.build_qwenvl_inputs()` 会在 enable_latent_reasoning 时尝试对齐 thinking token 位置（左 padding）。
   - analysis 时尽量走同一条路径，避免 batch 内 thinking token 位置错位导致切片错误。

3. **img_next tokens 与 thinking span 的边界**
   - 你的约定是 `<img_next>` 应该出现在末尾且连续（通常 16 个），避免后续还有文本。
   - 分析时请同时记录 img_next token span，避免把 img_next hidden 误当 latent。

4. **“分工”的定义要贴合你的格式**
   - 你当前 stage4 的默认顺序常见是 `[SUBTASK, BBOX, REASON]`（见 config 的 `component_order`），token 位置的语义依赖这个约定。
   - 如果后续你想改 `component_order` 或 `tag2think_count`，analysis 也要把该配置记录进 meta。

---

## 4) 最小可执行版本（MVP）建议

如果你希望先做“最快能出结论的一步”，建议只做：

1. Phase A：抽 200 条训练样本，确认 token 计数、位置、`num_reasoning_passes` 正确；
2. Phase B：做相似度分布 + effective rank（区分 token1/2/3）；
3. Phase C（轻量版）：按 suite 做 PCA（token1/2/3 各一张图）+ 一个线性 probe 预测 suite id。

做到这一步，你基本就能回答：
- latent 是否明显塌缩？
- latent 是否携带任务区分信息？
- 三个位置是否有“分工”趋势？

---

## 5) 记录与可复现性

每次分析建议把以下信息写入 `run_meta.json`：
- ckpt 路径、config（至少包含 `bridge_reasoning.stage/component_order/tag2think_count`、`latent_reasoning.*`、`img_next.*`）
- 数据源（dataset mix、suite/task 列表、样本数量、过滤规则）
- 抽样随机种子
- 是否使用 `forward_latent`、是否启用 img_next teacher 等
