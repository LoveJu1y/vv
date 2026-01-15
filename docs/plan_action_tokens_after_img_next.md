# Action Tokens After `<img_next>` (Keep Original Text Loss + Action Token Loss)

## Goal

We want to:

1. Keep the **existing language/text loss** behavior (the current `compute_language_loss` path).
2. Make **fast-tokenizer action tokens participate in CE loss** (important).
3. **Append action tokens after `<img_next>`**, and put them **at the very end** of the sequence.
4. **Do not require fixed action token length**: whatever the fast tokenizer outputs is used.
5. Avoid reusing the `solutions`-only workflow; instead **handle action tokens as a separate suffix** appended at token-level.

## High-level Scheme

- Dataset still produces `sample["action_tokens"]` (fast tokenizer).
- Formatter no longer injects `"Action: ..."` into the instruction text.
- `QwenGR00T` collects `action_tokens` from the batch and passes them into `QWen3.build_qwenvl_inputs(...)`.
- `QWen3.build_qwenvl_inputs(...)` / `_build_qwenvl_inputs_with_alignment(...)` will:
  1. Build the normal input ids for the text (including the existing `<img_next>` span).
  2. Tokenize each sample’s `action_tokens[i]` into ids.
  3. Optionally append a fixed `" Action:"` prefix (recommended to include a leading space as `" Action:"` so it is clearly separated from the `<img_next>` span).
  4. Append the action token ids to the end of each sample (after `<img_next>` and the prefix).
  5. Re-pad to batch.
  6. Construct `labels` so that:
     - The original “text loss” behavior stays the same.
     - `<img_next>` tokens stay masked out of CE.
     - The `" Action:"` prefix and **action token ids are NOT masked** (so they contribute to CE).

This works with variable action token lengths because padding already handles variable sequence lengths; labels will ignore padding but include action ids.

## Per-file Changes

### 1) `starVLA/dataloader/gr00t_lerobot/bridge_reasoning_formatter.py`

**Purpose:** stop mixing action tokens into natural language, to avoid unstable tokenization and to ensure action tokens are a clean suffix after `<img_next>`.

**Changes:**

- In `_format_latent(...)` (stage >= 2):
  - Remove/disable:
    - `action_text = f"Action: {action_tokens}"`
    - `text = f"{text} {img_span} {action_text}"`
  - Keep: `img_next` span insertion at the end.
  - Final formatted instruction should end with: `... <img_next><img_next>...` (no action text).

- If stage0/stage1 also embed `"Action: ..."` (currently stage>=2 does), remove it there too to keep behavior consistent.

**Result:** `sample["lang"]` / `sample["language"]` has no action tokens; the suffix will be appended later at token-level.

---

### 2) `starVLA/dataloader/gr00t_lerobot/datasets.py`

**Purpose:** keep generating `sample["action_tokens"]` with fast tokenizer.

**Changes (likely none):**

- Keep existing block that sets `sample["action_tokens"]` when:
  - `fast_tokenizer_name` is configured, and
  - `include_action_tokens` is enabled.

**Note / future option:**

- Current implementation uses `current_action = action[0]` (single-step). If we later want action tokens for a full horizon, that logic will need updating, but it is out-of-scope for the first iteration.

---

### 3) `starVLA/model/framework/QwenGR00T.py`

**Purpose:** pass action token strings from the batch into the VLM input builder.

**Changes:**

- In `forward(...)`:
  - Collect: `action_tokens = [ex.get("action_tokens", "") for ex in examples]`
  - Call:
    - `self.qwen_vl_interface.build_qwenvl_inputs(images=..., instructions=..., action_tokens=action_tokens)`

- In `predict_action(...)`:
  - Usually no ground-truth `action_tokens` exist.
  - Either pass `action_tokens=None` or omit the argument (implementation should allow both).

---

### 4) `starVLA/model/modules/vlm/QWen3.py`

**Purpose:** token-level append of action token ids after `<img_next>`, and ensure they participate in CE while keeping original text loss.

#### 4.1 `build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs)`

**Changes:**

- Accept `action_tokens: Optional[List[str]]` via `kwargs` or explicit param.
- After `apply_chat_template(... tokenize=True ...)` produces `input_ids/attention_mask`:
  - If `action_tokens` is provided:
    - For each sample `i`:
      - `prefix_ids = tokenizer.encode(\" Action:\", add_special_tokens=False)` (optional)
      - `action_ids_i = tokenizer.encode(action_tokens[i], add_special_tokens=False)`
      - Append `prefix_ids + action_ids_i` to the end of that sample’s `input_ids`
      - Append ones to `attention_mask`
    - Re-pad to batch tensors.

**Labels:**

- Keep original text loss logic:
  - For the ECOT/latent pipeline, the alignment path already uses `_build_ecot_labels_batch`.
  - For the normal (non-alignment) path, ensure label construction is consistent with the project’s intended behavior (may require invoking `_build_ecot_labels_batch` similarly when `compute_language_loss=true`).
- Keep the invariant:
  - Padding ids => `IGNORE_INDEX`
  - `<img_next>` ids => `IGNORE_INDEX`
  - **The `\" Action:\"` prefix and action ids are trainable (not masked)**.

**Truncation policy (important):**

- Because action tokens are appended at the end, naive truncation can drop them.
- We should pick a consistent truncation strategy so that action suffix is preserved whenever possible.

#### 4.2 `_build_qwenvl_inputs_with_alignment(...)`

**Changes:**

- After thinking-token alignment and re-batching:
  - Append action token ids per-sample (same approach).
  - Re-pad to batch.
  - Then build labels using `_build_ecot_labels_batch` so:
    - pre-thinking instruction masked as before,
    - thinking span masked as before,
    - `<img_next>` masked as before,
    - appended action tokens remain unmasked and contribute to CE.

---

## Config Notes (no new fields)

- Keep using:
  - `datasets.vla_data.bridge_reasoning.include_action_tokens: true`
  - `datasets.vla_data.bridge_annotations.fast_tokenizer_name: ...`

We do **not** add new config knobs for the first iteration; the meaning of `include_action_tokens` shifts from “string injection into prompt” to “token-level suffix + CE”.

## One Clarification to Confirm

When you say “原本文本的loss也要算”:

- If it means: keep current `compute_language_loss` behavior (mask instruction + thinking span, and compute CE on post-thinking text), then this plan matches.
- If you want to also compute CE on the instruction part (i.e., do not mask instruction), then `_build_ecot_labels_batch` policy needs to be changed (larger behavior change).

## Minimal Validation Checklist (post-implementation)

1. Decode a sample input sequence:
   - Ensure it ends with `<img_next>*16` then action token ids (no `Action:` prefix text).
2. Check `labels` mask:
   - `<img_next>` ids are `IGNORE_INDEX`
   - action token ids are not `IGNORE_INDEX`
   - padding is `IGNORE_INDEX`
3. Run a tiny forward pass / `py_compile` sanity checks on modified files.
