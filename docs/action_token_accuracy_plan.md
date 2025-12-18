## Bridge Action Token Accuracy 规划

为了在 Stage‑2/3/4 训练中更直观地观测“隐式动作 token” 学习情况，计划新增 `action_token_accuracy` 指标。该指标统计 VLM 输出的 `<robot_action_*>` token 与真实 token 的匹配程度，可用于判断 latent reasoning 是否在正确捕捉动作意图。

---

### 1. 定义

- **目标**：对 `qwen_vl_interface.forward_latent()`（或普通 `forward`）输出的 logits，取动作 token 段的 argmax，与 Ground Truth 的 action token 序列逐 token 对比，得到准确率。
- **Ground Truth 来源**：当前 step 的动作通过 `Fast_Action_Tokenizer.encoder_action2vlmtoken(actions)` 得到的 `<robot_action_*>` 序列；训练时我们已经将其写入 `sample["action_tokens"]`，并在 Stage 1 时显式拼入 `lang`。
- **讨论点**：
  1. 官方 tokenizer 会把 `<robot_action_* >` 映射到 2048 个连续索引（参考 `QWen3.py` 中 `_ACTION_TOKEN_MIN/MAX`）；因此可以用 `[labels == token]` 判断，并统计该区间内的 token 数。
  2. 如果在 `forward_latent` 中进行多次推理，则准确率可基于最终 pass 的 logits 计算。

---

### 2. 实现方案

1. **Collect positions**  
   - 在 `QWen3.py` 中已有 `action_token_min/max` 常量；在 `build_qwenvl_inputs` 得到 `input_ids` 后，可建立一个 mask `action_mask = (labels >= min) & (labels <= max)`。
   - 可借鉴 `_determine_latent_spans` 的方式，把 action token 区段提取出来。

2. **Compute accuracy**  
   - 在 `QwenGR00T.forward` 的 VLM loss 分支中（即拿到 `vlm_outputs.logits` 后），计算：
     ```python
     preds = logits.argmax(dim=-1)
     matches = ((preds == labels) & action_mask).sum()
     total = action_mask.sum()
     ```
   - 注意跳过 `IGNORE_INDEX`（-100）位置；可让 action_mask 本身避免  -100。

3. **Logging**  
   - 将 `action_token_accuracy = matches / (total + 1e-8)` 写入 `result`，并在 `train_ecot.py` 的日志更新中打印/写到 wandb。
   - 由于会因 batch 中 action token 数差异而抖动，建议用滑动平均或只在 main process 打印。

---

### 3. 验证流程

1. **单元测试**  
   - 可构造一个 dummy batch：`lang` 中包含 `<robot_action_0><robot_action_1>` 等；手动设置 logits，让部分匹配/部分不匹配，验证准确率计算。
2. **训练观察**  
   - Stage 1/2 的训练日志中，新增 `vlm_loss` 后面打印 `action_acc=xx.xx`；对 stage2+，预期准确率初期约等于随机猜测（接近 1/2048），随着训练上升。

---

### 4. 接口参数

在配置中可增加开关：

```yaml
trainer:
  log_action_token_accuracy: true
```

或在 `framework.latent_reasoning` 下附加 `enable_action_accuracy: true`。这样非 Bridge 任务可以关闭此计算，避免额外算力。
