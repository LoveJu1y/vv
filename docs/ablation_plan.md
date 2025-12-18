 ## 目标
- 支撑四组消融（无CoT、VLM见过CoT但推理不输出、显式CoT、隐式CoT）在同一 backbone/groot 上训练与评测，保持可维护、可复用。
- 训练与推理均用统一开关，最小化脚本分叉；耗时仅在评测侧记录。

## 配置与开关（建议）
- 统一模式枚举 `cot_mode ∈ {none, vlm_seen_no_out, explicit, implicit}`，落在配置的两个位置：
  - VLM/数据侧：`datasets.vla_data.bridge_reasoning.stage` 与 `framework.enable_latent_reasoning / latent_reasoning.*`。
  - 动作头侧：`framework.action_model` 下的 `use_reasoning_summary`、`use_reasoning_film`、`film_first_k`。
- 运行脚本参数化：
  - `COT_MODE` 环境变量驱动上述枚举（在 `run_starvla_bridge.sh` 增加 switch，根据模式注入对应 dotlist）。
  - 思维解码策略：仅对隐式/显式需要；显式用纯文本 CoT，可单独配置 `CO_TEXT_MAX_LEN`；隐式用 `THINK_MAX_LEN`、`THINK_TEMP`、`THINK_TOPP`。
  - 耗时仅在评测侧记录：分别统计思维生成与动作推理。

## 训练侧改动点
- `train_ecot.py`
  - 支持从 `cot_mode` 派生以下布尔：
    - `emit_thinking_tokens`（显式输出并参与 cross-attn）。
    - `latent_reasoning_enabled`（启用 thinking hidden → summarizer/FiLM）。
    - `vlm_seen_cot`（数据含 CoT，但推理可不输出）。
  - 将这些标志传入 `build_framework(cfg)`：可通过在 cfg.framework 下写入 `forward_mode` 或独立字段，保持向后兼容。
  - `prepare_data`：根据 `cot_mode` 选择数据混合（Stage0/Stage1）；为显式/隐式模式确保 dataloader 传出 `reasoning_mask`。
  - 日志/校验：`validate_ecot_config` 增加对 `cot_mode` 的提示；训练侧不需要新增耗时打点。
  - 训练-推理一致性：显式模式训练时，确保 forward 使用显式 thinking tokens 参与 cross-attn；隐式模式使用 summarizer/FiLM。
  - 实施细节建议：
    - 在 `main(cfg)` 读取 `cot_mode`（从 CLI/dotlist），生成派生布尔注入 `cfg.framework` 和 `cfg.datasets.vla_data`。
    - 如果 `cot_mode` 需要 Stage0/Stage1 切换，利用 `datasets.vla_data.dataset_py` 或 `data_mix` 切换不同数据清单；Stage1 对应含 CoT 数据。
    - 若模型需要显式 tokens：在 forward 时增加 `emit_thinking_tokens` 路径（文本拼接进入 cross-attn），并确保 dataloader 提供 `reasoning_mask`。
    - 若模型需要隐式 summarizer/FiLM：维持现有 `use_reasoning_summary`/`use_reasoning_film`，但由 `cot_mode` 统一驱动。

## 推理/评测改动点
- `examples/SimplerEnv/start_simpler_env.py`
  - 透传新的 `cot_mode`/思维解码超参到 `M1Inference`。
- `examples/SimplerEnv/model2simpler_interface.py`
  - 在 `M1Inference.__init__` 中增加 `cot_mode` 解析，派生：
    - 是否在 `_format_instruction_with_reasoning` 中添加 thinking span（仅隐式；显式为纯文本 CoT 不加思维 token；无/seen_no_out 不生成）。
    - 是否设置 `use_iterative_forward`（仅隐式需要）。
  - 若显式：自回归生成纯文本 CoT（无思维 tokens），再 forward 取 hidden 全量进 cross-attn；记录 `thinking_gen_time`。
  - 若隐式：自回归生成 thinking hidden，传入 summarizer/FiLM（不拼文本）；记录 `thinking_gen_time`。
  - 若 vlm_seen_no_out：不生成思维，`reasoning_mask=None`，保持原有 action 流程。
  - 若 none：完全关闭思维相关逻辑。
  - 保持 websocket 接口的 payload 兼容：新增字段可选，例如 `thinking_hidden`/`reasoning_mask` 或 `mode`，需和服务端约定。
  - 实施细节建议：
    - `M1Inference.__init__`：增加 `cot_mode`、`think_*` 超参；根据 mode 设置 `enable_latent_reasoning`、`emit_thinking_tokens`、`use_iterative_forward`。
    - 自回归生成路径（显式/隐式）：在 `step` 前或模型调用前，调用 VLM 生成函数，获取文本与 hidden_states 以及 `reasoning_mask`；将生成耗时计入返回日志。
  - 显式模式（纯文本 CoT，不含思维 tokens）：自回归生成 CoT 文本，再做一次 forward 取 hidden，全量进 cross-attn；无需 reasoning_mask。
  - 隐式模式：不拼文本，直接传 `thinking_hidden`（或让服务端根据 mask 聚合），驱动 summarizer/FiLM。
    - Websocket payload：新增可选字段 `cot_mode`、`thinking_hidden`、`reasoning_mask`、`thinking_text`（显式），保持旧字段兼容。

## run 脚本调整
- `scripts/run_starvla_bridge.sh`
  - 新增 `COT_MODE` 环境变量（none/vlm_seen_no_out/explicit/implicit）。
  - 基于模式注入 dotlist：
    - 数据/阶段：`datasets.vla_data.bridge_reasoning.stage`（none=0，vlm_seen_no_out=1，explicit=1，implicit=4），`datasets.vla_data.ecot.scheduled_stage` 同步。
    - 框架：explicit 关闭 latent reasoning/FiLM/summary；implicit 开启 latent reasoning，并按需开启 FiLM/summary。
    - 预训练：vlm_seen_no_out 使用 Stage1 VLM ckpt + Stage0 action；others 视实验设定。
  - 输出目录区分 `RUN_ID` 后缀（cot mode、数据量、显式/隐式）。
  - 实施细节建议：
    - 用 case/switch 根据 `COT_MODE` 追加不同的 dotlist 片段：
      - none：`enable_latent_reasoning=false`，stage=0，关闭 reasoning_summary/film。
      - vlm_seen_no_out：stage=1（VLM 见 CoT），`enable_latent_reasoning=false`，action 用 Stage0 数据。
      - explicit：stage=4，`enable_latent_reasoning=true`，`use_reasoning_summary=true`，可关闭 FiLM（或按需开启）。
      - implicit：stage=4，`enable_latent_reasoning=true`，`use_reasoning_film/summary` 按主线配置。
    - 增加思维解码超参（温度/top-p/max_len/stop_token）为可覆写的环境变量并下发到 cfg。

## 推荐的实验映射
- E1 none：Stage0 45k，`enable_latent_reasoning=False`，不生成思维。
- E2 vlm_seen_no_out：VLM Stage1 10k 预训，动作 Stage0 45k，推理不生成思维。
- E3 explicit：Stage1 45k 文本 CoT（无思维 tokens），推理显式自回归文本，再 forward 取 hidden 全量进 cross-attn。
- E4 implicit（当前主线）：Stage1 45k，推理自回归但仅用 thinking hidden → summarizer/FiLM。

## 兼容性与维护
- 不破坏现有默认路径：`cot_mode` 缺省为 `implicit`（保持当前行为）。
- 尽量用配置派生逻辑，不在训练循环中硬编码分支；推理端通过可选 payload 字段与服务端协商，避免 breaking change。
- 指标与耗时：评测记录思维生成耗时与动作推理耗时；训练侧无需新增耗时打点。

---

# 详细实施计划

## Phase 1: 配置与开关统一（1-2 小时）

### 1.1 新建配置解析工具
**文件**: `starVLA/training/trainer_utils/cot_mode_utils.py`（新建）

```python
from enum import Enum
from typing import Tuple

class CotMode(str, Enum):
    NONE = "none"                    # 无 CoT 数据，无思维生成
    VLM_SEEN_NO_OUT = "vlm_seen_no_out"  # VLM 见过 CoT，推理不输出
    EXPLICIT = "explicit"            # 显式自回归生成 + tokens 进 cross-attn
    IMPLICIT = "implicit"            # 隐式推理：hidden → summarizer/FiLM

def parse_cot_mode(cfg) -> CotMode:
    """从配置解析 cot_mode，缺省为 implicit（保持兼容）"""
    raw = cfg.framework.get("cot_mode", "implicit")
    return CotMode(raw.lower())

def derive_flags_from_mode(mode: CotMode) -> dict:
    """派生训练/推理所需布尔开关"""
    return {
        "enable_latent_reasoning": mode == CotMode.IMPLICIT,
        "emit_thinking_tokens": False if mode == CotMode.EXPLICIT else (mode == CotMode.IMPLICIT),
        "use_iterative_forward": mode == CotMode.IMPLICIT,
        "generate_thinking": mode in (CotMode.EXPLICIT, CotMode.IMPLICIT),
        "reasoning_stage": {
            CotMode.NONE: 0,
            CotMode.VLM_SEEN_NO_OUT: 1,
            CotMode.EXPLICIT: 1,
            CotMode.IMPLICIT: 4,
        }[mode],
    }
```

### 1.2 修改训练脚本支持 COT_MODE
**文件**: `scripts/run_starvla_bridge.sh`
**修改位置**: 第 47 行附近（参数定义区）

新增环境变量和 case 分支：
```bash
COT_MODE="${COT_MODE:-implicit}"

# 根据 COT_MODE 派生配置
case "${COT_MODE}" in
  none)
    SCHEDULED_STAGE=0
    ENABLE_LATENT_REASONING="false"
    USE_REASONING_FILM="false"
    USE_REASONING_SUMMARY="false"
    ;;
  vlm_seen_no_out)
    SCHEDULED_STAGE=1
    ENABLE_LATENT_REASONING="false"
    USE_REASONING_FILM="false"
    USE_REASONING_SUMMARY="false"
    ;;
  explicit)
    # 显式 CoT：不使用思维 tokens / FiLM，走纯文本 CoT
    SCHEDULED_STAGE=1          # 仅文本 CoT（无思维 token 对齐）
    ENABLE_LATENT_REASONING="false"
    USE_REASONING_FILM="false"
    USE_REASONING_SUMMARY="false"
    EMIT_THINKING_TOKENS="false"
    ;;
  implicit)
    # 当前主线配置
    SCHEDULED_STAGE=4
    ENABLE_LATENT_REASONING="true"
    USE_REASONING_FILM="true"
    USE_REASONING_SUMMARY="false"
    ;;
esac

# 追加到 TRAIN_CONFIG_ARGS
TRAIN_CONFIG_ARGS+=(
  --framework.cot_mode "${COT_MODE}"
  --framework.enable_latent_reasoning "${ENABLE_LATENT_REASONING}"
)
```

---

## Phase 2: 训练侧适配（2-3 小时）

### 2.1 train_ecot.py 引入 cot_mode
**文件**: `starVLA/training/train_ecot.py`
**修改位置**: `main(cfg)` 函数开头（第 618 行附近）

```python
from starVLA.training.trainer_utils.cot_mode_utils import parse_cot_mode, derive_flags_from_mode

def main(cfg) -> None:
    logger.info("ECoT VLA Training :: Warming Up")

    # === 新增：解析 cot_mode 并派生配置 ===
    cot_mode = parse_cot_mode(cfg)
    mode_flags = derive_flags_from_mode(cot_mode)
    
    # 注入派生配置到 cfg（保持向后兼容）
    cfg.framework.enable_latent_reasoning = mode_flags["enable_latent_reasoning"]
    cfg.framework.emit_thinking_tokens = mode_flags.get("emit_thinking_tokens", False)
    cfg.datasets.vla_data.bridge_reasoning.stage = mode_flags["reasoning_stage"]
    cfg.datasets.vla_data.ecot.scheduled_stage = mode_flags["reasoning_stage"]
    
    logger.info(f"[CotMode] mode={cot_mode.value}, flags={mode_flags}")
    # === 新增结束 ===

    sync_bridge_reasoning_to_framework(cfg)
    # ... 后续不变
```

### 2.2 数据混合切换
**修改位置**: `prepare_data` 函数或脚本层面

对于 `vlm_seen_no_out`，需要两阶段：
1. Stage1 VLM 预训练（见 CoT）→ 保存 checkpoint
2. Stage0 Action 训练（无 CoT）→ 加载上述 VLM checkpoint

脚本层面通过 `PRETRAINED_CKPT` 区分：
```bash
if [[ "${COT_MODE}" == "vlm_seen_no_out" ]]; then
  # 使用 Stage1 预训 VLM ckpt
  PRETRAINED_CKPT="${VLM_STAGE1_CKPT}"
  RELOAD_MODULES="qwen_vl_interface"
  # 数据用 Stage0（无 CoT）
  STEPS_CACHE_PATH="${STAGE0_STEPS_CACHE}"
fi
```

### 2.3 显式模式 forward 路径
**文件**: `starVLA/model/framework/QwenGR00T.py`
**修改位置**: `forward` 方法（第 90 行附近）

当 `emit_thinking_tokens=True` 时，需要把 thinking tokens 的 hidden states 也保留给 cross-attn（而非仅用 summarizer）：

```python
def forward(self, examples, **kwargs):
    # ... 现有代码 ...

    if cot_mode == "explicit":
        # 纯文本 CoT：不含思维 tokens，不用 forward_latent
        vlm_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        last_hidden = vlm_out.hidden_states[-1]
        reasoning_mask = None  # 全量 hidden 进 cross-attn
    elif use_iterative_forward:  # implicit
        vlm_outputs = self.qwen_vl_interface.forward_latent(...)
        last_hidden = vlm_outputs["hidden_states"]
        vlm_loss = vlm_outputs.get("loss")
        reasoning_mask = self._extract_reasoning_mask(qwen_inputs)
    else:  # none / vlm_seen_no_out
        vlm_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        last_hidden = vlm_out.hidden_states[-1]
        reasoning_mask = None

    # ... 后续 action_model forward ...
```

---

## Phase 3: 推理侧适配（3-4 小时）

### 3.1 评测入口透传参数
**文件**: `examples/SimplerEnv/start_simpler_env.py`
**修改位置**: 第 35 行附近

```python
from examples.SimplerEnv.custom_argparse import get_args
# ... 省略 ...

if __name__ == "__main__":
    args = get_args()
    # ... 省略 ...

    model = M1Inference(
        policy_ckpt_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,
        enable_latent_reasoning=args.enable_latent_reasoning,
        thinking_token_count=args.thinking_token_count,
        cot_mode=args.cot_mode,  # 新增
    )
```

**文件**: `examples/SimplerEnv/custom_argparse.py`（需确认是否已有）
新增参数：
```python
parser.add_argument("--cot_mode", type=str, default="implicit",
                    choices=["none", "vlm_seen_no_out", "explicit", "implicit"])
```

### 3.2 M1Inference 支持四种模式
**文件**: `examples/SimplerEnv/model2simpler_interface.py`
**修改位置**: `M1Inference.__init__`（第 491 行附近）

```python
class M1Inference:
    def __init__(
        self,
        # ... 现有参数 ...
        cot_mode: str = "implicit",  # 新增
    ) -> None:
        # ... 现有初始化 ...
        
        # === 新增：cot_mode 解析 ===
        self.cot_mode = cot_mode.lower()
        
        # 派生布尔开关
        self.generate_thinking = self.cot_mode in ("explicit", "implicit")
        self.emit_thinking_tokens = self.cot_mode == "explicit"
        self.use_iterative_forward = self.cot_mode in ("explicit", "implicit")
        
        # 覆盖/校正 enable_latent_reasoning
        if self.cot_mode in ("none", "vlm_seen_no_out"):
            self.enable_latent_reasoning = False
        else:
            self.enable_latent_reasoning = True
        
        # 耗时统计
        self.thinking_gen_times = []
        self.action_infer_times = []
```

### 3.3 step 方法适配
**文件**: `examples/SimplerEnv/model2simpler_interface.py`
**修改位置**: `step` 方法（第 628 行附近）

```python
import time

def step(self, image, task_description=None, *args, **kwargs):
    # ... 现有前置逻辑 ...
    
    # === 根据 cot_mode 构造 instruction ===
    if self.cot_mode == "none":
        instruction = self.task_description
        use_iterative = False
    elif self.cot_mode == "vlm_seen_no_out":
        instruction = self.task_description  # 不添加 thinking span
        use_iterative = False
    elif self.cot_mode in ("explicit", "implicit"):
        instruction = self._format_instruction_with_reasoning(self.task_description or "")
        use_iterative = True
    else:
        instruction = self.task_description
        use_iterative = False

    vla_input = {
        "batch_images": [[image]],
        "instructions": [instruction],
        "unnorm_key": self.unnorm_key,
        "do_sample": False,
        "cfg_scale": self.cfg_scale,
        "use_ddim": self.use_ddim,
        "num_ddim_steps": self.num_ddim_steps,
        "use_iterative_forward": use_iterative,
        "cot_mode": self.cot_mode,  # 透传给服务端
        "emit_thinking_tokens": self.emit_thinking_tokens,  # 显式模式标记
    }

    # === 记录推理耗时 ===
    t0 = time.perf_counter()
    response = self.client.infer(vla_input)
    t1 = time.perf_counter()
    
    # 分别记录（可从 response 解析）
    thinking_time = response.get("data", {}).get("thinking_gen_time", 0)
    self.thinking_gen_times.append(thinking_time)
    self.action_infer_times.append(t1 - t0 - thinking_time)
    
    # ... 后续 unnormalize 等不变 ...
```

### 3.4 服务端适配（可选）
**文件**: `deployment/model_server/tools/websocket_policy_server.py`
**修改位置**: `_route_message` 方法（第 96 行附近）

如果需要服务端返回 `thinking_gen_time`，在 `predict_action` 调用前后计时：

```python
elif mtype == "infer":
    # ... 现有校验 ...
    try:
        payload["batch_images"] = image_tools.to_pil_preserve(payload["batch_images"])
        
        # === 计时 ===
        import time
        t0 = time.perf_counter()
        output_dict = self._policy.predict_action(**payload)
        t1 = time.perf_counter()
        output_dict["total_infer_time"] = t1 - t0
        # thinking_gen_time 由 predict_action 内部返回（如需细分）
        
    except Exception as e:
        # ...
```

### 3.5 QwenGR00T.predict_action 适配
**文件**: `starVLA/model/framework/QwenGR00T.py`
**修改位置**: `predict_action` 方法（第 246 行附近）

```python
@torch.inference_mode()
def predict_action(
    self,
    batch_images,
    instructions,
    state=None,
    use_iterative_forward=False,
    cot_mode="implicit",  # 新增
    emit_thinking_tokens=False,  # 新增
    **kwargs,
):
    # ... 现有图像预处理 ...
    
    # 根据 cot_mode 决定 reasoning_mask 是否传递
    if cot_mode == "explicit" and emit_thinking_tokens:
        # 显式模式：不用 summarizer，全量 hidden 进 cross-attn
        reasoning_mask = None
    else:
        reasoning_mask = self._extract_reasoning_mask(qwen_inputs)
    
    # ... 后续调用 action_model.predict_action 不变 ...
```

---

## Phase 4: 自回归生成（显式 CoT 专用，4-5 小时）

### 4.1 设计思路
显式 CoT 模式下，推理时需要 **真正生成** 思维文本（而非用固定 placeholder tokens）。

当前隐式模式通过 `forward_latent` 迭代更新 thinking token embeddings（替换而非生成文本）。
显式模式需要：
1. 先让 VLM 自回归生成思维文本直到 `<|end_of_thinking|>`
2. 取生成过程的 hidden states
3. 将生成的 tokens（或 hidden）拼入 cross-attn

### 4.2 新增显式生成方法
**文件**: `starVLA/model/modules/vlm/QWen3.py`
**位置**: `_QWen3_VL_Interface` 类内（第 507 行附近）

```python
def generate_thinking_explicit(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    max_thinking_len: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs,
):
    """
    显式自回归生成思维文本，直到遇到 end_thinking_id 或达到最大长度。
    
    Returns:
        dict:
            - generated_ids: [B, L_gen] 生成的 token IDs
            - hidden_states: [B, L_total, H] 包含生成过程的 hidden states
            - thinking_text: List[str] 解码后的思维文本
            - gen_time: float 生成耗时（秒）
    """
    import time
    t0 = time.perf_counter()
    
    B = input_ids.shape[0]
    device = input_ids.device
    
    # 生成配置
    gen_config = {
        "max_new_tokens": max_thinking_len,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "eos_token_id": self.end_thinking_id,
        "pad_token_id": self.processor.tokenizer.pad_token_id,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **gen_config,
        )
    
    generated_ids = gen_output.sequences  # [B, L_prompt + L_gen]
    
    # 提取生成部分的 hidden states
    # gen_output.hidden_states 是 tuple of tuple，每步一个
    # 需要合并为 [B, L_total, H]
    all_hidden = []
    for step_hidden in gen_output.hidden_states:
        # step_hidden[-1] 是最后一层，shape [B, 1, H]（每步生成一个 token）
        all_hidden.append(step_hidden[-1])
    hidden_states = torch.cat(all_hidden, dim=1)  # [B, L_gen, H]
    
    # 解码文本
    thinking_texts = []
    for b in range(B):
        gen_tokens = generated_ids[b, input_ids.shape[1]:]
        text = self.processor.tokenizer.decode(gen_tokens, skip_special_tokens=False)
        thinking_texts.append(text)
    
    t1 = time.perf_counter()
    
    return {
        "generated_ids": generated_ids,
        "hidden_states": hidden_states,
        "thinking_text": thinking_texts,
        "gen_time": t1 - t0,
    }
```

### 4.3 QwenGR00T.predict_action 使用显式生成
**文件**: `starVLA/model/framework/QwenGR00T.py`
**修改位置**: `predict_action` 方法

```python
if cot_mode == "explicit":
    # 显式模式：自回归生成思维
    gen_result = self.qwen_vl_interface.generate_thinking_explicit(
        input_ids=qwen_inputs["input_ids"],
        attention_mask=qwen_inputs["attention_mask"],
        pixel_values=qwen_inputs.get("pixel_values"),
        image_grid_thw=qwen_inputs.get("image_grid_thw"),
        max_thinking_len=kwargs.get("max_thinking_len", 64),
    )
    
    # 用生成的完整序列（prompt + thinking）做一次 forward 获取 hidden
    full_ids = gen_result["generated_ids"]
    full_mask = torch.ones_like(full_ids)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vlm_out = self.qwen_vl_interface(
            input_ids=full_ids,
            attention_mask=full_mask,
            pixel_values=qwen_inputs.get("pixel_values"),
            image_grid_thw=qwen_inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
        )
    last_hidden = vlm_out.hidden_states[-1]
    reasoning_mask = None  # 显式模式全量进 cross-attn
    thinking_gen_time = gen_result["gen_time"]
    
elif cot_mode == "implicit" and use_iterative_forward:
    # 隐式模式：现有 forward_latent
    vlm_outputs = self.qwen_vl_interface.forward_latent(...)
    last_hidden = vlm_outputs['hidden_states']
    reasoning_mask = self._extract_reasoning_mask(qwen_inputs)
    thinking_gen_time = 0  # 隐式无额外生成耗时
    
else:
    # none / vlm_seen_no_out：普通 forward
    vlm_out = self.qwen_vl_interface(...)
    last_hidden = vlm_out.hidden_states[-1]
    reasoning_mask = None
    thinking_gen_time = 0
```

---

## Phase 5: 测试与验证（2-3 小时）

### 5.1 单元测试
**文件**: `starVLA/tests/test_cot_modes.py`（新建）

```python
import pytest
from starVLA.training.trainer_utils.cot_mode_utils import CotMode, parse_cot_mode, derive_flags_from_mode

def test_derive_flags():
    assert derive_flags_from_mode(CotMode.NONE)["enable_latent_reasoning"] == False
    assert derive_flags_from_mode(CotMode.IMPLICIT)["enable_latent_reasoning"] == True
    assert derive_flags_from_mode(CotMode.EXPLICIT)["emit_thinking_tokens"] == True
    assert derive_flags_from_mode(CotMode.IMPLICIT)["emit_thinking_tokens"] == False

def test_reasoning_stage():
    assert derive_flags_from_mode(CotMode.NONE)["reasoning_stage"] == 0
    assert derive_flags_from_mode(CotMode.VLM_SEEN_NO_OUT)["reasoning_stage"] == 1
    assert derive_flags_from_mode(CotMode.EXPLICIT)["reasoning_stage"] == 4
```

### 5.2 集成验收
运行四种模式的短训练 + SimplerEnv 评测，验证：
1. 训练 loss 正常下降
2. 评测指标与预期一致
3. 耗时统计输出正确

```bash
# E1: none
COT_MODE=none NUM_GPUS=1 MAX_TRAIN_STEPS=100 bash scripts/run_starvla_bridge.sh

# E2: vlm_seen_no_out (需先有 Stage1 VLM ckpt)
COT_MODE=vlm_seen_no_out PRETRAINED_CKPT=... bash scripts/run_starvla_bridge.sh

# E3: explicit
COT_MODE=explicit bash scripts/run_starvla_bridge.sh

# E4: implicit (当前主线)
COT_MODE=implicit bash scripts/run_starvla_bridge.sh
```

---

## 架构兼容性保障

1. **配置派生**：所有模式差异通过 `cot_mode_utils.py` 统一派生，避免硬编码分支。
2. **向后兼容**：缺省 `cot_mode=implicit`，现有脚本无需修改即可运行。
3. **Payload 兼容**：Websocket 新增字段为可选，旧客户端仍可正常工作。
4. **模块化**：显式生成逻辑独立为 `generate_thinking_explicit`，不影响隐式路径。
5. **测试覆盖**：新增单元测试确保派生逻辑正确。

---

## 实施顺序建议

| 阶段 | 内容 | 预估耗时 | 依赖 |
|------|------|----------|------|
| Phase 1 | 配置与开关统一 | 1-2h | 无 |
| Phase 2 | 训练侧适配 | 2-3h | Phase 1 |
| Phase 3 | 推理侧适配 | 3-4h | Phase 1 |
| Phase 4 | 显式生成实现 | 4-5h | Phase 2, 3 |
| Phase 5 | 测试与验证 | 2-3h | Phase 1-4 |

总计约 **12-17 小时**工作量，建议优先完成 Phase 1-3（支持 none/vlm_seen_no_out/implicit），Phase 4（显式）可后续迭代。
