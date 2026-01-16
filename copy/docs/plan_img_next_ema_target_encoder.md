# Plan: img_next EMA Target Vision Encoder（最小配置变更）

目标：在保持现有 `framework.img_next` 配置不扩展（仅 `enable/token/loss_weight/res`）的前提下，引入一个 **仅视觉 encoder 的 EMA teacher**，用于 `<img_next>` token 与下一帧视觉特征的对齐监督，降低自蒸馏闭环带来的表征退化风险，并保持实现改动面尽可能小。

## 现有实现快速回顾（已具备的链路）
- 文本侧：`BridgeReasoningFormatter` 会插入 `16 * "<img_next>"`（通常放在 action 前/无 action 末尾）。
- Tokenizer 侧：`starVLA/model/modules/vlm/QWen3.py` 会在 `img_next.enable=true` 时将 `<img_next>` 加入 tokenizer，记录 `img_next_token_id`，并在 labels 中将其 mask（不参与语言损失）。
- Loss 侧：`starVLA/model/framework/QwenGR00T.py::_compute_img_next_loss`
  - 从 `last_hidden` 抽取 `<img_next>` 位置的 `pred`（期望每样本 16 个 token）。
  - 对 `sample["image_next"]` 做视觉编码得到 `target_feats`（现为同一视觉 encoder 的 stop-grad 输出）。
  - 对末帧/缺失用 `image_next_fallback` mask 掉该样本的 loss。

## 设计原则
- **不增加 YAML 字段**：沿用现有 `framework.img_next.{enable, token, loss_weight, res}`。
- **Teacher 仅负责生成 target 特征**：不参与反传，不进 optimizer，仅 EMA 更新。
- **尽量复用现有接口**：保持 `QwenGR00T._compute_img_next_loss` 逻辑结构不大改（仍可先用 L1）。
- **`res` 真正生效**：下一帧图像在进入 processor/视觉 encoder 前，先 resize 到 `res x res`（默认 112）。

## 改动计划（最小实现）

### 1) 在 `QWen3.py` 内引入 teacher 视觉 encoder（EMA）
文件：`starVLA/model/modules/vlm/QWen3.py`
- 条件：仅当 `framework.img_next.enable=true` 时创建。
- 做法：
  - 从 student 的视觉子模块 `deepcopy` 得到 `vision_encoder_ema`（只拷视觉部分，避免复制整模型）。
  - 设置 `vision_encoder_ema.requires_grad_(False)` + `vision_encoder_ema.eval()`。
  - 提供一个 teacher 特征提取方法（名称可为 `get_image_features_target(...)`），输出与现有 `model.get_image_features(...)` 同结构，供下游直接调用。
- EMA 动量：先在代码内使用一个安全默认值（例如 `m=0.999`）；后续若需要可再开放为可选配置（本计划先不加）。

#### 代码修改 scheme（建议落点）
- 在 `QWen3Interface.__init__`（完成 `self.model/self.processor` 初始化、以及 `<img_next>` token 注入之后）：
  - 增加成员：
    - `self.enable_img_next = enable_img_next`（复用现有逻辑）
    - `self.img_next_ema_momentum = 0.999`（常量默认值）
    - `self.vision_encoder_ema = None`
  - 若 `enable_img_next`：
    - 从 student 视觉子模块构造 teacher：
      - `self.vision_encoder_ema = copy.deepcopy(self._get_student_vision_encoder())`
    - `self.vision_encoder_ema.requires_grad_(False); self.vision_encoder_ema.eval()`
- 新增 3 个方法（私有/公有均可）：
  1) `_get_student_vision_encoder(self) -> nn.Module`
     - 返回当前 Qwen3-VL 主模型里用于 `get_image_features` 的视觉 encoder 子模块。
     - 具体路径依赖 Qwen3-VL 实现（例如 `self.model.visual` / `self.model.vision_tower` 等），需在代码中做若干 `getattr` 兼容分支。
  2) `_get_teacher_vision_encoder(self) -> Optional[nn.Module]`
     - 返回 `self.vision_encoder_ema`。
  3) `update_img_next_ema(self, momentum: Optional[float]=None) -> None`
     - 在 `torch.no_grad()` 下执行 teacher 参数 EMA 更新：
       - `p_ema.mul_(m).add_(p_student, alpha=1-m)`
     - 在 `enable_img_next=False` 或 teacher 不存在时直接 return。

### 2) EMA 更新时机（optimizer.step() 之后）
文件：`starVLA/training/train_ecot.py`
- 在 `ECOTVLATrainer._train_step` 里：
  - `self.optimizer.step(); self.lr_scheduler.step()` 之后调用一次 EMA 更新方法，例如：
    - `self.model.qwen_vl_interface.update_img_next_ema()`（或 framework 转发到 interface）。
- 更新规则（无梯度）：
  - `ema = m*ema + (1-m)*student`
- 注意：
  - teacher 不参与梯度、不进入 optimizer param groups。
  - 兼容 accelerate/deepspeed：EMA 更新只进行参数拷贝/加权，不创建计算图。

#### 代码修改 scheme（建议落点）
- 在 `ECOTVLATrainer._train_step` 中：
  - `self.optimizer.step()` 与 `self.lr_scheduler.step()` 之后追加：
    - `if hasattr(self.model, "qwen_vl_interface") and hasattr(self.model.qwen_vl_interface, "update_img_next_ema"):\n    self.model.qwen_vl_interface.update_img_next_ema()`
  - 仅在 `self.accelerator.sync_gradients` 为 True 时更新（避免梯度累积阶段多次 EMA）：
    - `if self.accelerator.sync_gradients: ... update ...`

### 3) `QwenGR00T._compute_img_next_loss` 使用 teacher 特征做 target
文件：`starVLA/model/framework/QwenGR00T.py`
- 保持现有逻辑结构：
  - `pred`：来自 `<img_next>` token 对应的 `last_hidden`。
  - `target_feats`：改用 teacher encoder 的输出（`no_grad`）。
  - 继续使用 `image_next_fallback` mask 掉无效样本。
- `res=112` 的处理：
  - 在进入 processor 前，将 `image_next` 的 PIL resize 到 `(res,res)`（并保持当前 pipeline 的 processor 归一化逻辑）。

#### 代码修改 scheme（建议落点）
- 在 `_compute_img_next_loss` 的 “Encode next images” 段：
  - 在 flatten 后对每张 PIL 做 `img.resize((target_res, target_res))`（只针对 next，不影响 current 训练分辨率）。
  - target 特征提取改为优先走 teacher：
    - 若 `hasattr(self.qwen_vl_interface, "get_image_features_target")`：
      - `img_embeds, _ = self.qwen_vl_interface.get_image_features_target(pixel_values=..., image_grid_thw=...)`
    - 否则 fallback 到当前 student `main_model.get_image_features(...)`（保证向后兼容）。
- 建议顺手移除/降低频率的 `print(f"[img_next_loss] ...")`（避免训练 IO 瓶颈），改为 logger debug 或在 main rank 每 N 步打印一次（非必须但推荐）。

### 4) Checkpoint 行为（不额外改动）
- teacher encoder 作为 `nn.Module` 挂在模型里，默认会包含在 `state_dict` 中。
- 现有的 `accelerator.get_state_dict(self.model)` 保存 checkpoint 时会把 teacher 一并保存；恢复同理，无需单独文件。

## 最小验证（建议）
- 跑 10–20 steps：
  - `img_next_loss` 非空且稳定；
  - teacher 参数 `requires_grad=False`；
  - EMA 更新后 teacher 参数发生变化（但无梯度）。
- 对比 `res=112` vs `224` 的显存/速度（确认 `res` 生效）。
- 末帧样本 `image_next_fallback=True` 时，`img_next_loss` 对该样本应被 mask 掉。
