# Plan: B2 单次解码获取 current+next（torchvision_av）

目标：在 `LeRobotMixtureDataset -> get_step_data -> transforms -> _build_sample_from_data` 的现有链路下，对每个样本、每个 `video_key` **只解码一次视频**，同时得到当前帧与下一帧，缓解 `torchvision_av/pyav` 在多 worker 下的 CPU 内存压力与随机 OOM。

## 背景问题（现状）
- `LeRobotMixtureDataset.__getitem__` 会先执行 `dataset.get_step_data(...)` 读取视频帧（current）。
- 当前 `_build_sample_from_data(...)` 又额外调用 `get_video(...)` 读取 current 和 next，导致每个样本每个 view 至少 3 次解码（Mixture + 多 worker 时非常重）。
- `torchvision_av` 分支的解码本身更重，重复调用容易在运行一段时间后触发 `av.error.MemoryError: Cannot allocate memory`。

## B2 方案（推荐）
在 **`get_step_data` 阶段就一次性读取 current+next**，并把 next 帧随同 `data` 一起流入 transforms 与 `_build_sample_from_data`，从而 `_build_sample_from_data` 完全不再触发视频解码。

### 1) 新增取帧 helper（单次解码多 timestamp）
文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`

- 新增方法（私有即可）：
  - `get_video_by_step_indices(trajectory_id, key, step_indices: np.ndarray) -> np.ndarray`
- 功能：
  - 输入 `step_indices`（例如 `[base_index, next_index]`），内部 clamp 到合法范围；
  - 用 `self.curr_traj_data["timestamp"]` 映射成 `video_timestamp`；
  - 调用 `get_frames_by_timestamps(video_path, video_timestamp, backend=self.video_backend, kwargs=...)`；
  - 返回 `(len(step_indices), H, W, C)`。

### 2) 在 get_step_data 中对 video modality 做 B2 分支
文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`

- 在 `get_step_data(self, trajectory_id, base_index)` 中，针对每个 `video` key：
  - 计算 `next_index = min(base_index+1, traj_len-1)`；
  - 若 `next_index == base_index`：current/next 复用（无需额外读）；
  - 否则：调用一次 `get_video_by_step_indices(..., [base_index, next_index])` 得到两帧；
  - 将：
    - current 写入 `data[key]`（保持与原来一致的 shape/语义）；
    - next 写入一个新字段（避免被现有 transforms/视频 modality keys 自动处理）：
      - `data[f\"video_next.{sub_key}\"]`，其中 `sub_key = key.replace(\"video.\", \"\")`，同样的 shape 约定（与 current 一致）。

> 关键原则：**保留原来的 key 不变**，以免影响现有 transforms；next 帧使用新 key 避免冲突。

### 3) transforms 兼容性策略（重要）
- 如果 transforms 会遍历 `self.modality_keys["video"]`，它只会处理 current key，不会处理 `__next` key。
- 为保证 next 也得到同样的预处理（resize/normalize），推荐二选一：
  - A) 在 transforms 之后，在 `_build_sample_from_data` 中对 next 帧单独做与 current 相同的最小处理（例如 resize 到 224）；
  - B) 或者扩展 transforms：让它识别 `__next` key 并复用同样的 transform（改动更大）。

为保持最小改动，优先选 A：让 transforms 仍只处理 current；next 在 `_build_sample_from_data` 里只做 `Image.fromarray(...).resize((224,224))`，与 current 行为一致。

### 4) 修改 _build_sample_from_data：完全不再 get_video
文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`

- current：用 `data[video_key][0]`
- next：
  - 优先用 `data[f\"{video_key}__next\"][0]`；
  - 若不存在（兼容旧数据/旧 cache）：fallback 到 current，并标记 `image_next_fallback=True`。

这样 `_build_sample_from_data` 不再触发任何视频解码，只负责组装 PIL 图像列表。

### 5) 标记字段
- `sample["image_next"]`: 与 `sample["image"]` 同长度（每个 view 一张 PIL）
- `sample["image_next_fallback"]`: bool，末帧或缺失时为 True

### 6) 验证点（必须做）
- 对单个样本，打印 `get_video`/`get_frames_by_timestamps` 的调用次数，确认每个 `video_key` 只发生一次解码。
- 跑一个小 batch，确认：
  - `image` / `image_next` 维度一致；
  - fallback 样本不崩；
  - `torchvision_av` 多 worker 时不再在几分钟后 OOM。

## 预期效果
- 将每个样本每个 view 的解码次数从“≥3 次”降到“≈1 次（B2）”；
- 显著降低 DataLoader worker 的 CPU 内存压力，减少 PyAV `Cannot allocate memory`。
