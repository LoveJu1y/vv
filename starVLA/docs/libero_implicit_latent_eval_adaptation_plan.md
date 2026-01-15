# LIBERO 评测适配 Implicit Latent Reasoning（ECoT）计划

## 目标与约束

**目标**
- 在 `examples/LIBERO` 的评测中，确保推理阶段真实走到 **implicit latent reasoning** 路径：
  - 输入中包含 `<|start_of_thinking|> <|thinking|>... <|end_of_thinking|>` 的 **latent span**；
  - 输入末尾包含 `<img_next>` tokens（用于 img-next 对齐/或 action head 的 mask）；
  - RPC payload 显式触发 `QwenGR00T.predict_action(..., use_iterative_forward=True, cot_mode="implicit")`，从而走 `QWen3.forward_latent`。

**非目标（暂不做）**
- 不实现 `none/explicit/vlm_seen_no_out` 等其它 cot_mode 的完整评测兼容；LIBERO 侧先把 `implicit` 跑通即可。
- 不修改训练代码与模型结构；仅修改评测/推理输入构造与必要的 debug/日志。

**最小侵入原则**
- 修改尽量集中在 `examples/LIBERO`（client/eval）侧；
- server 侧只加“只读 debug/健康检查/更稳的退出回收”，不改核心推理逻辑；
- 默认行为尽量保持可回退（可以通过参数关闭 implicit 增强，便于对照）。

---

## 现状梳理：LIBERO 评测数据流（当前为什么可能没走 latent）

### 现有链路
1. `examples/LIBERO/eval_libero.py`
   - 从 LIBERO 环境拿到 `task_description + images`。
   - 调用 `examples/LIBERO/model2libero_interface.py:M1Inference.step(...)` 发起推理请求。
2. `examples/LIBERO/model2libero_interface.py`
   - 构造 websocket payload：`batch_images`, `instructions`, `unnorm_key`, `use_ddim/num_ddim_steps` 等。
   - 当前实现 **基本只发送纯 instruction**（缺少 thinking/img_next 结构，也不一定传 `use_iterative_forward/cot_mode`）。
3. `deployment/model_server/server_policy.py`
   - 加载 checkpoint -> `policy = baseframework.from_pretrained(ckpt)`（实际是 `QwenGR00T`）。
   - websocket 收到 payload -> `policy.predict_action(**payload)`。
4. `starVLA/model/framework/QwenGR00T.py:predict_action`
   - 是否走 latent 取决于：
     - `cot_mode == "implicit"` 且 `use_iterative_forward == True`；
     - 输入 token 序列中能被识别出 reasoning span（`reasoning_mask`）；
     - 如果使用 img_next：输入末尾包含 `<img_next>` tokens（用于 `img_next_mask`）。

### latent reasoning 被触发的关键条件（必须同时满足）
- **条件 A**：server 侧 `QwenGR00T.predict_action` 收到 `cot_mode="implicit"`，且 `use_iterative_forward=True`。
- **条件 B**：`instructions` 本身包含形如 `"... . @ <|start_of_thinking|><|thinking|>...<|end_of_thinking|> ..."` 的结构，使得 `QWen3.build_qwenvl_inputs` 能生成 `reasoning_mask`。
- **条件 C（可选但推荐对齐训练）**：`instructions` 末尾追加连续的 `<img_next>` tokens（且最好是末尾最后一段 token），以对齐 `QWen3` 的 img_next attention/mask 逻辑。

> 结论：如果 LIBERO client 只传“纯 instruction”，即使 checkpoint 开启 latent reasoning，也可能等价于 baseline 路径（mask 为空 / use_iterative_forward=False）。

---

## 参考实现：SimplerEnv 是怎么做的（我们要复用的模式）

`examples/SimplerEnv/model2simpler_interface.py` 已实现 implicit：
- 构造 prompt：`"{task}. @ {thinking_span} {img_next_span}"`（thinking/img_next token **无空格拼接**，与训练一致）。
- payload 除了 `instructions` 还会传：
  - `use_iterative_forward`（implicit 为 True）
  - `cot_mode="implicit"`
  - `emit_thinking_tokens`（implicit 通常 False）
  - `think_max_len/think_temp/think_topp`（显式模式生成用；implicit 可保留但不关键）

我们要在 LIBERO 侧“复刻这套输入构造 + payload flags”，先只支持 implicit。

---

## 修改计划（按最小侵入排序）

### Step 1：在 LIBERO client 增加 implicit prompt 构造（核心）
**文件**：`examples/LIBERO/model2libero_interface.py`

**改动点**
- 给 `M1Inference` 增加/补齐与 Simpler 对齐的参数（只保留 implicit 需要的最小集）：
  - `cot_mode`（默认 `"implicit"`，且只实现 implicit）
  - `use_iterative_forward`（implicit 强制 True，或由 `enable_latent_reasoning` 推导）
  - `thinking_token_count`（或“从 ckpt 自动推断的 total_thinking_tokens”）
  - `img_next_count`（默认 16，优先从 ckpt/config 自动推断）
  - thinking/img_next token 字符串（优先从 ckpt config 读：`framework.latent_reasoning.*` 和 `framework.img_next.token`）
- 复用 Simpler 的格式，新增一个私有函数：
  - `_format_instruction_with_latent(prompt: str) -> str`
  - **严格对齐格式（避免多余字符）**：
    - 使用训练侧 formatter 与 SimplerEnv 一致的分隔符：`. @ `（点号 + 空格 + `@` + 空格）。
    - thinking span 内部 **不允许** 插入任何空格/分隔符：`{start}{thinking*count}{end}`（纯 token 拼接）。
    - img_next span 内部 **不允许** 插入任何空格/分隔符：`{img_next_token * img_next_count}`（纯 token 拼接）。
    - thinking span 与 img_next span 之间仅保留 **一个空格**（与训练侧 `_append_img_next`/Simpler 一致）。
    - 最终建议返回：`f\"{prompt}. @ {start}{thinking*count}{end} {img_next*count}\".strip()`
    - 并确保 `<img_next>` 是末尾最后一段（`<img_next>` 后面不要再拼任何文本/标点/空格；`.strip()` 仅用于去掉意外的尾随空白）。
- 在 `step()` 构造 payload 时：
  - `instructions=[formatted_instruction]`
  - 追加字段：
    - `use_iterative_forward=True`
    - `cot_mode="implicit"`
    - `emit_thinking_tokens=false`（implicit）

**为什么最小侵入**
- 不动 server，不动模型；只改变发过去的文本与 kwargs。

**成功判据**
- server 日志能看到 `[ECOT] Completed ... reasoning passes in predict_action`（`QwenGR00T.predict_action` 已有该 log）。
- 或者能在返回 payload 中看到 `thinking_gen_time`（implicit 一般为 0，但 `total_infer_time` 会存在；重点是 server 侧能确认走 forward_latent）。

---

### Step 2：token 数量与字符串自动对齐 checkpoint（避免“瞎填 token”）
**文件**：`examples/LIBERO/model2libero_interface.py`

**动机**
- 训练时 thinking token 数量可能来自 `bridge_reasoning.stage + tag2think_count`（例如 stage4 => SUBTASK+BBOX+REASON 三段，合计 3 个 `<|thinking|>`）。
- img_next token 数量通常是 16（与 `res=112` 对应 2x2 merge 后 16 tokens），但仍应以 config 为准。

**策略**
- 在 `__init__` 里已通过 `read_mode_config(policy_ckpt_path)` 读取到 `model_config`：
  - 读取 thinking tokens 字符串：
    - `framework.latent_reasoning.start_of_thinking_token`
    - `framework.latent_reasoning.thinking_token`
    - `framework.latent_reasoning.end_of_thinking_token`
  - 读取 img_next token 字符串：
    - `framework.img_next.token`（默认 `<img_next>`）
  - 读取 stage 与 tag2think_count：
    - 优先：`datasets.vla_data.bridge_reasoning.stage` + `datasets.vla_data.bridge_reasoning.tag2think_count`
    - 退化：`thinking_token_count` 使用默认值（与训练时保持一致的默认：如 3 或 4，需要你确认训练配置习惯）
  - 读取 img_next_count：
    - 优先：`datasets.vla_data.bridge_reasoning.img_next_count`（如果存在）
    - 否则默认 16

**输出对齐日志（只打印一次）**
- 打印：最终使用的 `thinking_token_count/img_next_count` 和 token 字符串，便于复现实验。

---

### Step 3：解决 multi-dataset 训练导致的 unnorm_key 问题（libero_all 必需）
**文件**：`examples/LIBERO/eval_libero.py` + `examples/LIBERO/model2libero_interface.py`

**现状风险**
- `M1Inference.get_action_stats()` 在 `unnorm_key is None` 且 `norm_stats` 有多个 key 时会直接 assert 失败。
- `libero_all` 训练往往产生多个 stats key（goal/object/spatial/libero_10）。

**方案**
- 在 `examples/LIBERO/eval_libero.py:Args` 增加 `unnorm_key: str`（可选）：
  - 推荐由 `task_suite_name` 自动映射到正确 key：
    - `libero_goal` -> `libero_goal_no_noops_1.0.0_lerobot`
    - `libero_object` -> `libero_object_no_noops_1.0.0_lerobot`
    - `libero_spatial` -> `libero_spatial_no_noops_1.0.0_lerobot`
    - `libero_10` -> `libero_10_no_noops_1.0.0_lerobot`
  - 或者用户显式传入 `--args.unnorm_key ...`
- 将 `unnorm_key` 传入 `M1Inference(...)`，并同时传给 server payload（如果 server 也需要）。

---

### Step 4：LIBERO eval 侧新增最小参数，保证“可控 + 可回归”
**文件**：`examples/LIBERO/eval_libero.py`

新增/补齐 args（只为 implicit 服务）：
- `enable_latent_reasoning: bool`（默认 True）
- `thinking_token_count: int`（默认 -1 表示自动对齐 ckpt；或直接不暴露，由 client 自动计算）
- `img_next_count: int`（默认 -1/16）
- `cot_mode: str`（默认 implicit，且仅允许 implicit）
- `unnorm_key: str`（见 Step 3）

目的：
- 不用改代码即可从命令行切换“baseline vs implicit”（例如把 `enable_latent_reasoning=false` 做对照实验）。

---

### Step 5：Server 侧加非侵入式可观测性（可选，但强烈建议）
**文件**：`deployment/model_server/tools/websocket_policy_server.py` 或 `deployment/model_server/server_policy.py`

仅增加 debug（通过环境变量开关，比如 `EVAL_DEBUG=1`）：
- 打印每次请求的：
  - instruction 前 120 字符；
  - 字符串级统计 `<|thinking|>` 出现次数、`<img_next>` 出现次数；
  - `use_iterative_forward/cot_mode` 是否按预期传入；
  - 返回中透传 `thinking_gen_time` / `total_infer_time`（目前 server 已追加 `total_infer_time`）。

目的：避免“以为走 latent，但其实 payload 没带/没被解析”。

---

### Step 6：更新/加一个 job 脚本示例（把 implicit eval 作为默认路径）
**文件**：`examples/LIBERO/eval_libero_job.sh`

建议增强（不影响核心逻辑）：
- server readiness：不要固定 `sleep 15`，改成循环探测端口可连（失败则退出）。
- `trap`：脚本异常退出时也能 kill server，避免 server 残留占端口。
- 对每个 task suite 传入：
  - `--args.cot_mode implicit`
  - `--args.enable_latent_reasoning true`
  - `--args.thinking_token_count ...`（可选）
  - `--args.img_next_count ...`（可选）
  - `--args.unnorm_key ...`（按 suite 自动映射）

---

## 验证步骤（建议按顺序做）

1. **最小连通性**
   - 启动 server（同一 ckpt）+ 跑 `libero_goal`，`num_trials_per_task=1`，确认能跑完 1 个 episode。
2. **确认 implicit 生效**
   - 开启 debug（client 打印最终 instruction；server 打印 token count / use_iterative_forward）。
   - 观察 server 是否出现 `forward_latent` 相关日志（例如 `Completed ... reasoning passes`）。
3. **对照实验**
   - 只改一个开关：`enable_latent_reasoning=false`（或不拼 token + use_iterative_forward=false），其余不变，对比 success/速度/行为。
4. **libero_all 四套件**
   - 跑 `libero_goal/spatial/object/libero_10` 各 1~2 个 episode，重点检查：
     - `unnorm_key` 是否正确（不会触发 assert；动作幅度合理）。
     - 只有 libero_10 有 bbox2 并不影响 eval（eval 不用 bbox）。

---

## 风险与对策

- **checkpoint 没有启用 thinking/img_next special tokens**
  - 现象：tokenizer 无这些 token id；或者 `<img_next>` 被当成普通字符。
  - 对策：在 client 初始化时读取 config，若 `framework.enable_latent_reasoning` / `framework.img_next.enable` 为 false，则直接警告并退回 baseline（或强制退出）。

- **thinking token 数量与训练不一致**
  - 现象：`reasoning_mask` 形状与预期不匹配，可能影响 action head 的 reasoning conditioning。
  - 对策：默认从 ckpt config 自动计算 token 数量（Step 2）；必要时允许 CLI 覆写。

- **multi-dataset unnorm stats 选错**
  - 现象：动作尺度异常，success 大幅下降。
  - 对策：按 task_suite_name 映射 unnorm_key（Step 3），并在启动时打印选择的 key。

---

## 交付物清单（完成后应该看到什么）

- `examples/LIBERO` 的评测在 `implicit` 下能稳定触发 `forward_latent`；
- `eval_libero_job.sh` 默认跑 4 套件时，每个套件都有：
  - 评测日志（stdout/可选 log_path）；
  - server 自动回收，不留端口占用；
- 对照实验（implicit vs baseline）只需要改 CLI 参数即可复现。
