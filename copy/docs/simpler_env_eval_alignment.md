## SimplerEnv 评测与训练数据构建的一致性检查

### 评测侧逻辑（SimplerEnv）
- `start_simpler_env.py` 调用 `M1Inference`，通过 WebSocket 请求后端推理，`vla_input` 中只包含：
  - 单张图片 `[[image]]`（`uint8`），先用 `cv2.resize` 到 `image_size`（默认 224×224，BILINEAR）。
  - 指令字符串：`task_description`，如果启用 latent reasoning 则拼接 `" @ " + thinking_sequence`。
  - `use_iterative_forward = enable_latent_reasoning`。
- `thinking_sequence` 构造方式：`" <|start_of_thinking|>" + "<|thinking|>" * thinking_token_count + "<|end_of_thinking|>"`，默认 `thinking_token_count=4`。
- 预测只取 `response["data"]["normalized_actions"][0]`，随后用 `dataset_statistics.json` 做反归一化；`state` 从未传入 action head（始终 `None`）。

### 训练侧关键路径（/starVLA/dataloader/gr00t_lerobot）
- 数据构造：`bridge_reasoning_formatter.py`  
  - Stage >=2 会把指定的组件（BBOX/SUBTASK/REASON/ACTION）替换成 thinking span。  
  - thinking span 精确格式（注意空格与句点）：  
    - 若有 instruction：`"{instruction}. @ {<|start|>...<|end|>}"`。instruction 末尾有句点，“ @ ” 两侧各一空格，thinking span 内部无空格。  
    - `thinking_body` 顺序按已有段的顺序追加（BBOX→SUBTASK→REASON→ACTION），每段贡献的 `<|thinking|>` 个数由 `tag2think_count` 决定（默认各 1，Stage4 常见总数=3，无 Action 时）。  
  - `include_action_tokens` 控制是否保留 Action 片段；使用“无 action token”权重时应设为 `false`。
- 训练脚本（如 `run_starvla_bridge.sh` action-only 模式）：`framework.training_stage="action_only"`，`bridge_reasoning.stage=4`，`include_action_tokens=false`。  
- `train_ecot.py` 的推理/训练接口默认使用 `processor.apply_chat_template`，期望指令结尾包含 `.@` 及 thinking span，无多余空格；token 计数依据 `tag2think_count`。

### 不一致/风险点
1) **Thinking token 数量**：  
   - 训练 Stage4 默认每个 TAG 1 个 `<|thinking|>`，BBOX+SUBTASK+REASON 共 3 个。  
   - 评测默认 `thinking_token_count=4`，总数 4，且没有区分不同 TAG。存在长度与分布不匹配风险。
2) **指令格式/分隔符**：  
   - 训练 `format_latent` 生成的字符串是 `"{instruction}. @ {<|start|>...<|end|>}"`（instruction 末尾含 `.`）。  
   - 评测侧直接 `instruction.strip() + " @ " + thinking_sequence.strip()`，无末尾句点，且 thinking_sequence 前有前导空格，易造成 tokenizer 序列差异。
3) **Action token 处理**：  
   - 评测期望“无 action token”，但需确保训练时 `include_action_tokens=false` 与 stage 对齐；否则权重学习到的分布与推理输入不匹配。  
   - `use_iterative_forward` 仅由 `enable_latent_reasoning` 控制，若未开 latent，会走单段前向，与训练 stage4（需要 latent 走迭代）不一致。
4) **图像预处理**：  
   - 评测用 `cv2.resize` + BILINEAR，训练侧使用 PIL BILINEAR（`ecot_rlds/transforms`）。数值非常接近，但仍有微弱差异；可选保持 PIL 路径一致。
5) **Thinking token 位置/顺序**：  
   - 训练根据存在的段（bbox→subtask→reason→action）顺序拼接 thinking body；评测侧固定一段连续 thinking，不携带段顺序信息。

### 建议对齐方案
1) **对齐 thinking token 模式**：  
   - 在评测端根据训练配置 `tag2think_count` 与 `bridge_reasoning.stage` 生成 thinking span：按 formatter 的段顺序累加 `<|thinking|>`，默认 Stage4（无 Action）总数=3。  
   - 保持完全一致的文本格式：`f\"{instruction.strip()}. @ <|start_of_thinking|>{body}<|end_of_thinking|>\"`，注意 instruction 末尾句点、“ @ ”两侧空格，span 内无空格。
2) **确保使用迭代前向**：  
   - Stage4 需要 `use_iterative_forward=True`；若禁用 latent，请显式设置并提醒可能性能下降。
3) **禁用 action tokens**：  
   - 保证启动脚本/服务端推理使用 `include_action_tokens=false` 的 checkpoint；评测输入无需添加任何 Action 文本。
4) **图像预处理一致性**（可选）：  
   - 将评测的 resize 改为 PIL BILINEAR（与训练完全一致），或确认 cv2 BILINEAR 误差可接受。

### 后续可修改点（供参考）
- 在 `model2simpler_interface.py` 中封装一个与 `BridgeReasoningFormatter` 一致的 thinking 序列构造函数，读取 checkpoint 的 `tag2think_count` / `bridge_reasoning.stage` 自动生成，避免手工设定 `thinking_token_count`。  
- 在 WebSocket 请求中增加一个开关以显式声明 `include_action_tokens=false`，与服务端加载的权重一致。  
- 若需要状态输入，需在推理侧补充 `state`，但当前模型/数据均未使用，可保持 `None`。
