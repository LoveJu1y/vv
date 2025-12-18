# BRIDGE CoT + Fast Action Token 训练规划

> 目标：在现有的 BRIDGE-LeRobot 数据加载与 `QwenGR00T`/`train_ecot.py` 流程上，借鉴 `QwenFast` 的思路，将当前 step 的 action 通过 fast tokenizer 离散化后，直接拼接到语言 prompt 的末尾，使其与 CoT/Reasoning 一起进入 VLM，从而在语言损失中监督动作分布。

---

## 1. 需求理解

1. **动作 token 化方式**  
   - 复用 `QwenFast` 中 `action_model.encoder_action2vlmtoken()` / `fast_tokenizer` 的离散化逻辑，将连续动作转换为 token 序列（string）。
   - 每个样本追加 “Action Tokens: <fast_token_seq>” 之类的段落，与 CoT 文本保持统一格式。

2. **插入位置**  
   - 放置在当前 formatter 输出的最后，形如：  
     ```
     Instruction ... @ Subtask ... Reasoning ... BBox ... ActionTokens: <fast tokens>
     ```
   - Stage ≥2 场景下，需要确保动作 token 段落是否进入 latent span：初期先保持显式（即不被 thinking token 覆盖），后续可按阶段需求决定是否 latent 化。

3. **训练影响**  
   - Stage 0/1/2/3/4 暂时都保持动作 token 显式存在（不加入 latent span），因此无需修改 `_determine_latent_tags_for_stage`；只要有文本即可被语言损失监督。
   - `forward_latent` 只负责 mask 掉 instruction/thinking span，动作 token 段落位于 thinking span 之后，天然会被纳入语言 loss。
   - 需要评估额外 token 长度对显存和序列长度的影响（特别是 fast token 序列长度与动作维度相关）。

---

## 2. 参考实现（QwenFast）

1. `starVLA/model/framework/QwenFast.py`
   - 在 `forward` 中调用 `self.action_model.encoder_action2vlmtoken(actions)` 得到 `vlm_action_tokens`，作为 `build_qwenvl_inputs(..., solutions=vlm_action_tokens)` 的输入。
   - 「动作 → token」的编码器定义在 `starVLA/model/modules/action_model/fast_ActionHeader.py`，内部持有 fast tokenizer。

2. `fast tokenizer` 的关键点：
   - `self.action_model.fast_tokenizer.time_horizon`、`action_dim` 等配置需要与 LeRobot action 维度一致。
   - 输出通常是字符串序列，便于直接拼到 prompt。

---

## 3. 实现步骤规划

1. **评估差异**
   - 确认 BRIDGE 动作维度/范围与 `StarVLA/Qwen3-VL-4B-Instruct-Action` 中 fast tokenizer 预期一致，必要时在配置中补充 normalization/clip。
   - 梳理 `LeRobotSingleDataset` 返回的 `action` 格式，确保能提取“当前 step 的动作”作为 fast tokenizer 输入。

2. **新增动作 token 生成**
   - 在 dataloader（`LeRobotSingleDataset._build_sample_from_data`）中调用 fast tokenizer，只对「当前 step 的动作」进行编码。
   - 将编码结果保存为 `sample["action_tokens"] = "<robot_action_x>..."`，供 formatter 使用。

3. **扩展 Formatter**
   - `BridgeReasoningFormatter` 在 `_format_stage1` / `_format_latent` 尾部新增 “ActionTokens: …” 段落，固定显式形式。
   - 为确保 `_find_ecot_spans` 能定位 instruction 段，可考虑在动作段落前增加固定前缀（如 `Action:`），并保证其位于 thinking span 之后。

4. **训练脚本对齐**
   - 确认 `bridge_reasoning.stage>=1` 时的 `forward_latent()` 会把动作 token 段落包含在语言损失中（目前只要它存在文本里即可）。  
   - 若未来希望动作 token 也纳入 latent span，需要扩展 `_determine_latent_tags_for_stage` 与 mask 逻辑。

5. **配置与开关**
   - 在 YAML 中新增 `bridge_reasoning.include_action_tokens`（默认 True），以及 fast tokenizer 相关路径/参数（如 `fast_tokenizer_model = "StarVLA/Qwen3-VL-4B-Instruct-Action"`）。
   - 保证 action encoder 只处理当前 step，所以需在 config 中指出动作维度、归一化方式，确保 fast tokenizer 与 BRIDGE 数据对齐。

6. **验证与测试**
   - 单元测试：给定 mock action，验证 formatter 输出包含正确的 token 序列。
   - 训练冒烟：Stage 1（显式）和 Stage ≥2（latent）各跑少量 step，确认 `forward_latent` 未报错、语言损失正常下降。
   - 序列长度检查：统计追加动作 token 后的平均长度，若超出 `model_max_length` 需同步调整。

---

## 4. 开放问题

1. fast tokenizer 权重来自 `StarVLA/Qwen3-VL-4B-Instruct-Action`，需确认下载/缓存流程（HF mirror、cache dir），并在 README 中记录使用方法。
2. 追加动作 token 是否会影响 diffusion action head 的学习？需观察 action loss 与语言 loss 的 trade-off。
3. 动作 token 暂不进入 latent span；若未来计划 latent 化，需要定义新的 stage（如 Stage 5）和 mask 策略。

---

## 5. 最终交付物

1. `BridgeReasoningFormatter` 支持拼接 fast action tokens，并可配置开关。
2. Dataloader/Batched sample 中包含 fast token 字段，保证与 formatter 对接。
3. 配置模板与脚本示例，说明如何启用该特性。
4. 测试脚本或文档，证明新增 token 不破坏现有训练流程。
