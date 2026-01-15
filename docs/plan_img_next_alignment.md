## img_next 隐式对齐任务改造计划（更新版）

目标：在 Stage1/2/3/4 的 VLM 前向中，为 motion reasoning 之后新增 16 个 `<img_next>` token，对齐下一帧视觉特征，使用 L1 监督。

### 已确认关键点
- Qwen3-VL vision encoder 内部有 2×2 特征 merge：224×224 → 64 token；112×112 → 16 token（无需额外池化）。
- 只使用下一帧的主视角进行对齐（不扩展到多视角）。
- `<img_next>` 为显式 token，不进入 thinking token 的迭代更新。
- Stage 位置与隐式化规则：  
  - Stage1：全显式 CoT，motion reasoning 之后追加 16×`<img_next>`（显式）。  
  - Stage2：第 1 个 CoT 组件隐式化，其余按原逻辑，随后 16×`<img_next>` 显式。  
  - Stage3：前 2 个 CoT 组件隐式化，其余显式，随后 16×`<img_next>` 显式。  
  - Stage4：3 个 CoT 组件全隐式，随后 16×`<img_next>` 显式。  
  - `<img_next>` 永远显式，固定在 motion reasoning 后、action 前。

### 待解决/注意点
- Tokenizer 扩展：新增 `<img_next>` special token，确保 embedding 初始化（可沿用其他 special token 初始化或均值初始化）。
- 序列插入位置：固定在 motion reasoning 之后、action 之前；Stage1/2/3/4 均保持 `<img_next>` 显式。
- 数据提供：dataloader 需返回 `image_next`（主视角），若缺失则 fallback 到当前帧但不计入 loss。
- Loss 权重：`total_loss = action_loss + α*vlm_loss + β*img_next_loss`；β 需配置化，默认可 0.5~1.0，需实验调节。  
  - `reasoning_only`：有 image_next 时可算 img_next_loss。  
  - `action_only`：默认跳过 img_next_loss（可配置覆盖）。  
- 评估/日志：训练路径计算 img_next_loss；推理接口默认不做下一帧对齐，只做可选调试日志；记录 fallback 计数/比例。

### 代码修改步骤（详细）
1) 数据 & Token 准备  
   - dataloader: 样本新增 `image_next`（主视角 PIL / tensor），transform 与当前视角一致（含归一化）。  
   - tokenizer: 注册 `<img_next>`，在 Qwen3 接口记录 `img_next_token_id`，确保不会被认为是 thinking token。

2) 序列构造  
   - `BridgeReasoningFormatter`: 在 motion reasoning 之后插入 16×`<img_next>`，Stage1/2/3/4 均显式保留。  
   - 保持 thinking tokens 逻辑不变；`<img_next>` 不参与 latent thinking。

3) 前向计算（QwenGR00T.forward）  
   - 取 `image_next` → resize 112×112 → 送入 qwen3 vision encoder，得到 16×D 特征（已确认）。  
   - 在 `last_hidden` 中用 `<img_next>` mask 抽取 16×D，与 encoder 输出做 L1 → `img_next_loss`。  
   - 融合总损失：`action_loss + vlm_loss_weight*vlm_loss + img_next_loss_weight*img_next_loss`（视阶段/配置决定是否包含）。  
   - forward_latent 兼容：保持 `<img_next>` 仅在最终 hidden_states 使用，不影响 thinking token 的迭代。  
   - 缺失 `image_next` 的 fallback：构造 batch 级 `use_fallback_mask`（[B]），缺失样本用当前帧代替编码；计算 img_next_loss 时对 fallback 样本置零/不计入均值，并记录 fallback 计数。

4) 推理路径（predict_action）  
   - 默认不做下一帧对齐，仅保留正常动作预测；可加调试标志打印 `<img_next>` 命中情况，但不计算 loss。

5) 配置与校验  
   - 新增：`framework.img_next.enable`, `img_next_loss_weight`, `img_next_token`, `img_next_res`(默认 112)。  
   - 在 config 校验中提示：若 enable=True 且未提供 `image_next`，则跳过并报警；分辨率必须为 112×112（否则需显式池化处理）。
   - 训练阶段开关：`reasoning_only` 可选算 img_next_loss（若有 vlm_loss）；`action_only` 建议默认跳过（配置可覆盖）。

6) 测试/冒烟  
   - 构造伪 batch：含 `image`, `image_next`, `lang`, `action`，验证 `<img_next>` mask 抽取形状为 [B,16,D]，vision encoder 输出为 [B,16,D]，L1 正常。  
   - 覆盖 Stage1/2/3/4 序列构造，确认 `<img_next>` 未被 thinking tokens 覆盖。  
   - 缺失 `image_next` 时应跳过 img_next_loss、不报错。  
   - fallback 路径：伪造部分样本缺少 `image_next`，检查 `use_fallback_mask`、loss 仅对有效样本求均值、fallback 计数上报。

### 风险与备选
- 训练耗时上升：vision encoder 额外前向，可选关闭梯度或共享视觉前处理缓存。  
- 数据缺失：需降级路径（跳过 img_next_loss）。  
- 权重敏感：β 需调参；可先 0.5 试验，再网格/自适应调整。

---

## 代码修改 Scheme（逐文件）

### 1) Token & 配置层
- `starVLA/model/modules/vlm/QWen3.py`
  - 在 tokenizer 注册 `<img_next>` special token，暴露 `img_next_token_id`。
  - 确保与 thinking token 判定解耦（避免被 forward_latent 处理）。
- `config`（YAML/omegaconf）
  - 新增 `framework.img_next.{enable, loss_weight, token, res}`，默认 `enable=False`, `loss_weight=0.5`, `res=112`。
  - `framework.cot_mode_flags` 无需改动；`cot_mode` 保持。

### 2) 数据层
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
  - 样本增加 `image_next`（主视角）；transform 复用当前视角流程，支持缺失降级（缺失则标记为 None）。
- `starVLA/dataloader/gr00t_lerobot/bridge_reasoning_formatter.py`
  - 在 motion reasoning 后插入 16×`<img_next>`（Stage1/2/3/4 均显式）。
  - 确保不与 thinking token 混淆；保持 component_order 稳定。

### 3) 模型前向
- `starVLA/model/framework/QwenGR00T.py`
  - 读取 `image_next`，resize 至 112×112（用现有 `resize_images` 或新增 util）。
  - 调用 Qwen3 vision encoder 获得 16×D 特征；若 encoder 输出非 16，显式池化到 16。
  - 从 `last_hidden` 用 `<img_next>` mask 抽取 16×D，对齐计算 `img_next_loss = L1`。
  - 总损失：`action_loss + vlm_w*vlm_loss + img_w*img_next_loss`，受阶段与配置控制：
    - `reasoning_only`: 若 enable 且有 image_next，则加 img_next_loss（与 vlm_loss 并存）。
    - `action_only`: 默认跳过 img_next_loss（可由配置覆盖）。
    - `full`: 正常加权。
  - forward_latent 保持不处理 `<img_next>`，仅最终 hidden 使用。
  - predict_action 不计算 img_next_loss，仅可选日志 `<img_next>` 命中。

### 4) 训练脚本 & 校验
- `starVLA/training/train_ecot.py`
  - 读取新配置，传递给框架。
  - 在 `validate_ecot_config` 增加 img_next 校验（enable→需 res=112 且模型已注入 token；缺失 image_next 时 warning 并降级）。

### 5) 测试与示例
- 新增/更新冒烟脚本（可在 `docs/` 或 `scripts/`）：
  - 构造伪 batch（含 image_next）跑一次 forward，断言 `<img_next>` mask 形状 [B,16,D]，L1 正常。
  - 覆盖 Stage1/2/3/4 序列插入检查。

### 6) 迭代与性能
- 如训练耗时增加：可在 img_next 分支关闭 vision encoder grad（只用于监督对齐），或缓存同批次的视觉特征。
- 权重调参：初始 `img_next_loss_weight=0.5`，与 `vlm_loss_weight` 并行网格。


