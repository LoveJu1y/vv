## SimplerEnv 评测对齐改造计划

目标：让 SimplerEnv 推理输入与训练时的 `gr00t_lerobot` 数据格式完全一致（Stage4、无 Action token），避免因指令格式 / thinking tokens 差异导致分布偏移。

### 改动点概览
1. **指令与 thinking span 生成**  
   - 在推理侧（`examples/SimplerEnv/model2simpler_interface.py`）封装一个函数按训练规则生成文本：先添加与训练一致的 base prompt  
     `"Robot task reasoning: first output the target bbox, then list the subtask, then generate the motion reasoning. Instruction: {instruction}"`，  
     再拼接 thinking span：`"{prompt}. @ <|start_of_thinking|>{body}<|end_of_thinking|>"`，其中 `body` 按训练时的段顺序累加 `<|thinking|>`（默认 Stage4、无 Action => 3 个）。  
   - 去掉自定义 `thinking_token_count` 的自由度，优先从 checkpoint/config 读取 `bridge_reasoning.stage`、`tag2think_count`、`include_action_tokens`，若读取失败才回退默认（Stage4，tag2think_count=1/1/1，include_action_tokens=false）。

2. **启用迭代前向**  
   - 确保 `use_iterative_forward=True` 当 `bridge_reasoning.stage >= 2`（尤其 Stage4）。在推理输入中显式设置，不依赖命令行开关。

3. **Action token 处理**  
   - 评测使用的 checkpoint 确认是 `include_action_tokens=false` 训练出的；推理端不添加任何 Action 文本。  
   - 若未来需要带 Action token 的权重，可通过配置读取判断是否应在 span 中增加 ACTION 段并计入 thinking body。

4. **指令尾部与空格**  
   - 保持训练格式：instruction 末尾加句点；`" @ "` 两侧各一空格；thinking span 内部无空格。

5. **图像预处理一致性（可选）**  
   - 将推理的 resize 改为 PIL BILINEAR，与训练一致；如保留 `cv2.resize`，需评估影响。

6. **配置读取**  
   - 复用 `read_config_simple` 或补充读取 `config.yaml` 中的 `datasets.vla_data.bridge_reasoning` 字段，自动导出 `stage`/`tag2think_count`/`include_action_tokens`，减少手工参数。

### 步骤拆解
1. 在 `model2simpler_interface.py`：
   - 添加从 checkpoint/config 解析 `bridge_reasoning` 的函数，返回 `stage`、`tag2think_count`、`include_action_tokens`，设置默认值（stage=4, tag2think_count={BBOX:1,SUBTASK:1,REASON:1}, include_action_tokens=false）。
   - 实现 `build_thinking_span(instruction, stage, tag2think_count, include_action_tokens)`，严格按 `BridgeReasoningFormatter` 的逻辑：按段顺序累加 thinking_token；格式 `"{instr}. @ <|start_of_thinking|>{body}<|end_of_thinking|>"`。
   - 在 `step` 中用上述函数替换现有 `thinking_sequence` 拼接；同时强制 `use_iterative_forward = (stage >= 2)`。

2. （可选）将图像 resize 换成 PIL BILINEAR，与训练一致。

3. 验证：  
   - 打印一次生成的指令样例，确认与 `BridgeReasoningFormatter.format` 在 Stage4、无 Action 时一致（3 个 `<|thinking|>`，句点+空格正确）。  
   - 用与训练相同的 checkpoint 跑一小批推理，确认没有 tokenizer 长度或视觉 token 数量问题。
