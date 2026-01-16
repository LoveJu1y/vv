## Action Head 强化方案（基于 QwenGR00T Stage-4）
针对当前 Stage-4 结构（VLM 固定、仅 3 个 latent thinking token），我们重点优化 `<|thinking|>` → Action Head 的利用。建议分三层推进，可按需叠加。

### 1. 显式提取 / 增强 latent 表示
- **Reasoning Encoder**：在 `QwenGR00T.forward` / `predict_action` 中，先用 `latent_mask` 取出 `<|thinking|>` span，通过轻量 Transformer/Attention 获得 `reasoning_summary`。可支持多尺度（forward_latent 每轮输出 / 多层 hidden state）汇聚。
- **与 hidden state 融合**：将 `reasoning_summary` 复制并拼到 latent 位置的 hidden state，或直接作为额外的 prefix token，保证动作头显式读取到推理摘要。

### 2. Action Head 内部条件化
- **FiLM/Gating**：在 DiT 各层引入 `Reasoning FiLM`，用 `reasoning_summary` 生成缩放/偏置（或 cross-attn query）调制视觉 token → action 的映射。
- **Residual Steering**：让 DiT 输出 baseline action，再由 `reasoning_summary` 预测一个 residual，`action = base + residual`，等价于“思考负责修正动作”。
- **条件 Query 扩展**：为 DiT 输入增加若干 learnable reasoning query，初始化为 latent hidden state，在整个动作生成过程中持续与视觉 token 交互。

### 3. 训练信号与流程
- **Auxiliary Loss（按资源择优）**：若要在 `reasoning_summary` 上加 bbox/subtask/trajectory 等辅助监督，最好在 VLM/forward_latent 阶段就引入。资源有限时，可以按以下梯度由低到高分层：  
  1. **低成本**：利用 Stage2/3 已有的 `bbox_valid`、`subtask_tag` 等离散标签，给对应 latent token 加一个简单分类或对比损失，成本接近 cross-entropy。  
  2. **中成本**：在 Stage1/2 中保留短句（如“Subtask: grasp carrot”），只取其短 embedding 作为 teacher，让 latent token 通过 MSE/InfoNCE 对齐，而不是完整生成文本。  
  3. **高成本**：若资源允许，再做完整的文本重构或 bbox 回归。  
  核心原则是这些监督要发生在 encoder 端，才能确保 latent token 在被 action head 读取前就具备语义；若只在 action head 端加 loss 而 VLM 冻结，效果有限。
- **分阶段 Fine-tune**：冻结 VLM，只调 action head + reasoning adapter；或先 freeze DiT 主体，只训 reasoning→FiLM 层，再逐步放开整体 action head，确保“如何使用 latent”成为训练重点。
- **Curriculum**：可先在 Stage1（显式 CoT）训练 action head 读取结构化文本，再迁移到 Stage4 latent+adapter 模式，减小直接从三个 `<|thinking|>` 开始的难度。

通过以上结构与训练策略，即使不修改 VLM，也能放大少量 latent token 对动作预测的影响，减少其在长序列中被淹没的问题。
