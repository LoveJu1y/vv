# Latent FiLM 调制方案（QwenGR00T + Action DiT）

## 背景
- 现状：`ReasoningSummarizer` 既生成 summary token 拼接到 DiT，又可能用于 FiLM，存在表征重复与逻辑不清的问题。
-,目标：聚焦 FiLM 路径，让 latent 作为全局调制信号；拼接路径可选/默认关闭，避免重复表征。

## 改动概要
1) **配置开关与模式**
   - 新增 `use_reasoning_film`（默认 false），控制是否对 DiT 层做 FiLM 调制。
   - 可选：`use_reasoning_summary_tokens`（默认 false），保留显式拼接的旧逻辑；两者可互斥或同时允许（默认仅 FiLM）。
   - FiLM 范围：`film_first_k`（如 4），`film_dropout`，`film_hidden_dim`（MLP 中间维）。

2) **简化/分流 summary**
   - 若 `use_reasoning_film` 为主路：直接用原始 latent（或投影+LN）做 pooling/加权 -> 得到一个 summary 向量，仅用于 FiLM，不再拼接 token。
   - 若保留 token 拼接，作为可选路径，默认关闭。

3) **FiLM 调制实现（Action DiT 内部）**
   - 在 `FlowmatchingActionHead` 初始化时，新增 `ReasoningFiLM`（MLP: latent_summary -> scale/shift）。
   - 在 `DiT` 的 block（或包装）里支持传入 `modulation`，对前 K 层做 AdaLN 风格：`x = LN(x) * (1 + scale) + shift`；无 modulation 时跳过。
   - 维度：`scale/shift` 与 DiT hidden_dim 对齐，dtype 跟随 hidden_states。

4) **数据流 & 兼容**
   - `QwenGR00T` 继续生成 reasoning_mask；动作头收到 `vl_embs`+`mask`。
   - 若 `use_reasoning_film=true` 且 mask 非空：计算 latent_summary -> modulation；传给 DiT。
   - 若 `use_reasoning_summary_tokens=true`：投影 latent token（或 summary token）拼到 future tokens；默认 false。
   - 默认配置保持旧行为（两者均 false）。

5) **配置/脚本**
   - `bridge_lerobot_stage2.yaml` 等：新增上述开关（默认 false），不改默认行为。
   - `run_starvla_bridge.sh`/`run_bridge_multistage.sh`：增加环境变量透传。

6) **校验与风险**
   - 混精：确保 FiLM MLP 与 LN 使用与 DiT 一致的 dtype，避免 BF16/FP32 mismatch。
   - 无 mask 容错：mask 为空直接跳过 FiLM/拼接。
   - 显存：FiLM MLP 很轻；关闭 token 拼接可减少重复存储。

## 交付顺序
1. 增加配置与脚本透传（默认关闭）。  
2. 在 ActionHead/DiT 中接入 FiLM 通路，保留旧路径但默认关。  
3. 本地最小验证（形状/混精），如 `py_compile` + 伪数据前向。  
4. 文档更新：注明如何开启/关闭 FiLM 与 token 拼接。***

---

## 代码修改计划（草案，先不实现）
1) **配置与默认值**
   - 在 config（如 `bridge_lerobot_stage2.yaml`）新增 `use_reasoning_film` 开关（默认 false），`film_first_k`、`film_dropout`、`film_hidden_dim`。
   - `use_reasoning_summary_tokens` 默认 false（可选移除 token 拼接路径或保留但默认关）。
   - 更新 `run_starvla_bridge.sh`、`run_bridge_multistage.sh` 透传上述开关。

2) **Summarizer 分流**
   - Latent 投影 + LN；用双独立 query 聚合出两路 summary（分别偏向 scale / shift），作为 FiLM 唯一输入；默认不再拼接 token。
   - 如需显式 token 拼接，保留开关（默认关），避免与 FiLM 重复表征。

3) **FiLM 模块**
   - 新增 `ReasoningFiLM`（小型 MLP），输入两路 summary -> `scale/shift`；`scale` 初始 0，支持 gate/温度、dropout。
   - 在 ActionHead 中生成 modulation，随 forward/predict 传入 DiT；无 summary/关闭开关时返回 None。

4) **DiT 接入**
   - 在 `cross_attention_dit.py` 的 block 中支持可选 modulation（AdaLN 风格），对前 `film_first_k` 层生效。
   - 确保 dtype 对齐（BF16/FP32），无 modulation 时走原路径。

5) **验证与回退**
   - 伪数据前向（训练/推理）验证形状/dtype；`py_compile` 检查。
   - 开关关闭时行为与旧版一致；若出现问题，可快速禁用 FiLM/拼接。
