# Latent Summarizer + Action Head Conditioning 计划

## 目标
在 **QwenGR00T → FlowmatchingActionHead** 流水线中，构建一条可靠的“latent reasoning summary → DiT 条件”路径，让 3 个 `<|thinking|>` latent token 在动作预测阶段被充分利用，同时保持现有 VLM、数据格式与训练脚本不变。

---

## 步骤概览
1. **定位与抽取 latent token**
   - 在 `QwenGR00T.forward` / `predict_action` 中记录 VLM 输出序列中 reasoning latent 的起止位置（训练/推理需一致）。
   - 将 latent slice 随 batch 一起传给 action head（如 `extra_inputs["reasoning_latents"]`），保持 FP16/BF16。

2. **设计 Latent Summarizer 模块**
   - 在 `FlowmatchingActionHead.__init__` 中新增 `LatentSummarizer` 子模块（可选 Transformer/MLP）。
   - 输入形状 `[B, N_latent, H_vlm]`，输出 `[B, N_summary, H_dit]`：  
     - 先线性映射到 `dit.cross_attention_dim`。  
     - 通过 1～2 层轻量 Transformer（Self-Attn + FFN）提炼语义。  
     - 最终压缩成 1～2 个 summary token（如均值 + learnable query 聚合）。

3. **与 state / future tokens 融合**
   - 将 summary token 重复/拼接到现有 `future_tokens` + `action_features` 拼接流：  
     - 方案 A：把 summary 作为新的“先验 token”，在 `future_tokens` 之前拼接，让 DiT 首先看到 reasoning 语义。  
     - 方案 B：将 summary 类比 state，进入 `state_features` 路径，通过 cross-attn 影响动作生成。
   - 优先实现方案 A（拼接到 future tokens），后续可开关式支持 B。
   - 对应添加配置 `use_reasoning_summary`, `summary_token_count` 等。

4. **训练/推理贯通**
   - `FlowmatchingActionHead.forward` 与 `predict_action` 中都要调用 summarizer，确保 eval 与训练一致。
   - 若缺少 latent（旧 ckpt / 特殊场景），需回退到旧逻辑（直接跳过 summarizer）。

5. **可选增强**
   - 为 summary 提供 DropPath/LayerNorm 以防 overfit。  
   - 加 `latent_summary_loss` 针对某些 proxy target（例如 bbox/subtask 标签），可在配置中默认关闭。
   - 记录 summary 激活（stats hook）便于调试。

---

## 里程碑 & 交付
1. **实现阶段**  
   - PR1：实现 latent 提取 + summarizer + DiT 输入融合，添加配置项，确认向后兼容。  
   - PR2（如时间允许）：加入可选辅助损失或 FiLM 调制。
2. **验证阶段**  
   - 在 Bridge Stage4 action-only 训练上对比旧版 action head，指标：train loss、libero/simpler eval 成功率。  
   - 记录 summary token 的范数/方差，确保未退化。
3. **文档**  
   - 更新 `docs/latent_action_head_improvements.md`，描述新模块与使用方式。

---

## 风险与对策
- **缺失 latent 数据**：若某些数据不含 reasoning token，summarizer 接口需容错，自动 bypass。  
- **维度错位**：VLM `hidden_size` 与 DiT cross-attn 维度不同，需显式线性投影；并在配置中声明。  
- **收敛不稳定**：在 summarizer 后加 LayerNorm/Residual，并提供关闭开关。

---

## 下一步
1. 明确 latent token 索引获取方式（解析 prompt 编码规则或在 dataloader 打标）。  
2. 在 `FlowmatchingActionHead` 中落地 summarizer + 接口。  
3. 编写单元测试/最小脚本，验证训练与推理路径都能跑通。***
