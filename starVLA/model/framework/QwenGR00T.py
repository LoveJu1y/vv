# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import hashlib
import json
import os
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # Training stage control: "reasoning_only", "action_only", or "full"
        self.training_stage = config.framework.get("training_stage", "full")
        self.use_reasoning_summary = getattr(self.config.framework.action_model, "use_reasoning_summary", False)
        self.use_reasoning_film = getattr(self.config.framework.action_model, "use_reasoning_film", False)
        
        # Apply parameter freezing based on training stage
        if self.training_stage == "reasoning_only":
            print(f"🔒 [Training Stage] reasoning_only mode - Freezing action_model parameters")
            for param in self.action_model.parameters():
                param.requires_grad = False
        elif self.training_stage == "action_only":
            print(f"🔒 [Training Stage] action_only mode - Freezing VLM parameters")
            for param in self.qwen_vl_interface.parameters():
                param.requires_grad = False
        else:
            print(f"🔓 [Training Stage] full mode - All parameters trainable")
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        action_tokens = [example.get("action_tokens", "") for example in examples]
        # img_next: List of PIL list (primary view), fallback flags
        image_next = [example.get("image_next", None) for example in examples]
        image_next_fallback = torch.tensor(
            [bool(example.get("image_next_fallback", False)) for example in examples],
            device=self.qwen_vl_interface.model.device,
        )
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format (tokenization and thinking token alignment if enabled)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            action_tokens=action_tokens,
        )
        reasoning_mask = (
            self._extract_reasoning_mask(qwen_inputs)
            if self.training_stage != "reasoning_only"
            else None
        )
        
        # Check if iterative implicit reasoning is enabled
        cot_mode = getattr(self.config.framework, "cot_mode", "implicit")
        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        use_iterative_forward = (
            cot_mode == "implicit"
            and enable_latent_reasoning
            and hasattr(self.qwen_vl_interface, "forward_latent")
        )
        if cot_mode == "explicit":
            # 显式 CoT：纯文本 forward，全量 hidden 进 cross-attn，不依赖 latent
            reasoning_mask = None
        
        if use_iterative_forward:
            # Step 2: Iterative forward with KV-Cache for implicit reasoning
            vlm_outputs = self.qwen_vl_interface.forward_latent(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                labels=qwen_inputs.get("labels"),  # May contain masked labels
                position_ids=qwen_inputs.get("position_ids"),
            )
            
            last_hidden = vlm_outputs['hidden_states']  # [B, L, H]
            vlm_loss = vlm_outputs.get('loss')  # May be None if no labels
            self._maybe_log_latent_analysis(
                qwen_inputs=qwen_inputs,
                last_hidden=last_hidden,
                vlm_outputs=vlm_outputs,
                instructions=instructions,
                use_iterative_forward=True,
                **kwargs,
            )
        else:
            # Step 2: Normal forward pass (no iterative reasoning)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
                vlm_loss = qwenvl_outputs.loss if hasattr(qwenvl_outputs, 'loss') else None
            self._maybe_log_latent_analysis(
                qwen_inputs=qwen_inputs,
                last_hidden=last_hidden,
                vlm_outputs=None,
                instructions=instructions,
                use_iterative_forward=False,
                **kwargs,
            )

        # Step 3: Compute losses based on training stage
        result = {}

        # 预计算 img_next_loss（reasoning_only / full 需要；action_only 默认跳过）
        img_next_loss = None
        img_next_cfg = getattr(self.config.framework, "img_next", {}) if hasattr(self.config, "framework") else {}
        enable_img_next = img_next_cfg.get("enable", False)
        img_next_loss_weight = img_next_cfg.get("loss_weight", 0.5)
        img_next_res = img_next_cfg.get("res", 112)
        img_next_token_id = getattr(self.qwen_vl_interface, "img_next_token_id", None)

        use_img_next_teacher = img_next_cfg.get("use_teacher", True)
        img_next_mask_for_action = (
            (qwen_inputs["input_ids"] == img_next_token_id) if img_next_token_id is not None else None
        )

        if (
            enable_img_next
            and use_img_next_teacher
            and img_next_token_id is not None
            and img_next_loss_weight is not None
            and img_next_loss_weight > 0
        ):
            img_next_mask = (qwen_inputs["input_ids"] == img_next_token_id)
            try:
                img_next_loss = self._compute_img_next_loss(
                    last_hidden,
                    image_next,
                    img_next_mask,
                    image_next_fallback,
                    target_res=img_next_res,
                )
            except Exception as e:
                logger.warning(f"[img_next_loss] skipped due to error: {e}")
                img_next_loss = None
        
        if self.training_stage == "reasoning_only":
            # Stage 1: Only train VLM reasoning, skip action head
            if vlm_loss is None:
                raise ValueError(
                    "training_stage='reasoning_only' requires VLM loss, but vlm_loss is None. "
                    "Please ensure enable_latent_reasoning=True and labels are provided."
                )
            result["vlm_loss"] = vlm_loss
            if img_next_loss is not None:
                result["img_next_loss"] = img_next_loss
                result["total_loss"] = vlm_loss + img_next_loss_weight * img_next_loss
            else:
                result["total_loss"] = vlm_loss
            return result

        elif self.training_stage == "action_only":
            # action_only mode: Only train action head, VLM is frozen
            with torch.autocast("cuda", dtype=torch.float32):
                # 标签对齐：取最后 chunk_len 段
                actions = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                
                state_repeated = None
                if state is not None:
                    state = torch.tensor(
                        np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                    )  # [B, state_dim] or [B, 1, state_dim]
                    
                    # Ensure state is 3D: [B, 1, state_dim]
                    if state.ndim == 2:
                        state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
                    
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)  # [B*repeated_diffusion_steps, 1, state_dim]

                reasoning_mask_repeated = self._repeat_reasoning_mask(reasoning_mask, repeated_diffusion_steps)
                img_next_mask_repeated = (
                    img_next_mask_for_action.repeat(repeated_diffusion_steps, 1)
                    if img_next_mask_for_action is not None
                    else None
                )
                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                    reasoning_mask=reasoning_mask_repeated,
                    img_next_mask=img_next_mask_repeated,
                )

                result["action_loss"] = action_loss
                result["total_loss"] = action_loss  # Only action loss
                if vlm_loss is not None:
                    result["vlm_loss"] = vlm_loss
                return result
        else:
            # full mode: Train both VLM and action head
            with torch.autocast("cuda", dtype=torch.float32):
                # 标签对齐：取最后 chunk_len 段
                actions = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                
                state_repeated = None
                if state is not None:
                    state = torch.tensor(
                        np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                    )  # [B, state_dim] or [B, 1, state_dim]
                    
                    # Ensure state is 3D: [B, 1, state_dim]
                    if state.ndim == 2:
                        state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
                    
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)  # [B*repeated_diffusion_steps, 1, state_dim]

                reasoning_mask_repeated = self._repeat_reasoning_mask(reasoning_mask, repeated_diffusion_steps)
                img_next_mask_repeated = (
                    img_next_mask_for_action.repeat(repeated_diffusion_steps, 1)
                    if img_next_mask_for_action is not None
                    else None
                )
                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                    reasoning_mask=reasoning_mask_repeated,
                    img_next_mask=img_next_mask_repeated,
                )

            result["action_loss"] = action_loss
            
            # Combine with VLM loss if available
        if vlm_loss is not None:
            vlm_loss_weight = self.config.framework.get("latent_reasoning", {}).get("vlm_loss_weight", 0.5)
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = action_loss + vlm_loss_weight * vlm_loss
        else:
            result["total_loss"] = action_loss

        # img_next 对齐损失（full/action_only 阶段，在上方预计算后合并）
        if (
            img_next_loss is not None
            and enable_img_next
            and use_img_next_teacher
            and img_next_loss_weight > 0
        ):
            result["img_next_loss"] = img_next_loss
            result["total_loss"] = result["total_loss"] + img_next_loss_weight * img_next_loss

        return result

    def _get_latent_analysis_cfg(self) -> dict:
        """
        Read config for lightweight latent-token analysis. Defaults to disabled.

        Supported config locations:
          - cfg.framework.latent_analysis
          - cfg.trainer.latent_analysis
        """
        cached = getattr(self, "_latent_analysis_cfg_cache", None)
        if isinstance(cached, dict):
            return cached

        cfg = self.config
        if cfg is None:
            return {}

        def _to_dict(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return dict(obj)
            # OmegaConf containers behave like attr objects; try getattr + iteration fallback.
            try:
                if hasattr(obj, "items"):
                    return dict(obj.items())
            except Exception:
                pass
            try:
                return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
            except Exception:
                return {}

        fw = getattr(cfg, "framework", None)
        tr = getattr(cfg, "trainer", None)
        fw_cfg = _to_dict(getattr(fw, "latent_analysis", None)) if fw is not None else {}
        tr_cfg = _to_dict(getattr(tr, "latent_analysis", None)) if tr is not None else {}

        merged = {}
        merged.update(fw_cfg)
        merged.update(tr_cfg)
        # Cache for future forward calls (config is effectively static during a run).
        self._latent_analysis_cfg_cache = merged
        return merged

    def _maybe_log_latent_analysis(
        self,
        qwen_inputs: dict,
        last_hidden: torch.Tensor,
        vlm_outputs: Optional[dict],
        instructions: List[str],
        use_iterative_forward: bool,
        **kwargs,
    ) -> None:
        """
        Lightweight analysis hook: extract thinking/img_next hidden states and dump summary stats.

        This is intentionally low-overhead and should be gated by config + step interval.
        """
        cfg = self._get_latent_analysis_cfg()
        if not cfg or not bool(cfg.get("enable", False)):
            return

        # ---- trigger / throttling ----
        # We support two ways to schedule analysis:
        #  1) External `global_step` passed in via kwargs (optional)
        #  2) Internal forward-call counter (default), no trainer changes needed.
        global_step = kwargs.get("global_step", None)

        # When trainer uses gradient accumulation, it may prefer skipping non-sync microsteps.
        # If the flag is not passed, we default to logging based on forward-call scheduling.
        sync_gradients = bool(kwargs.get("analysis_sync_gradients", True))

        interval = int(cfg.get("interval_steps", 0) or 0)
        if interval <= 0:
            # Backward-compatible alias: allow interval_forwards
            interval = int(cfg.get("interval_forwards", 0) or 0)
        if interval <= 0:
            return

        # Determine whether we are allowed to write (rank0 only).
        is_main_process = kwargs.get("is_main_process", None)
        if is_main_process is None:
            try:
                is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            except Exception:
                is_main_process = True
        is_main_process = bool(is_main_process)
        if not is_main_process:
            return

        # If a real global_step is provided, use it; otherwise, fallback to internal counter.
        if global_step is None:
            if not hasattr(self, "_latent_analysis_forward_calls"):
                self._latent_analysis_forward_calls = 0
            self._latent_analysis_forward_calls += 1
            global_step = int(self._latent_analysis_forward_calls)

        if (not sync_gradients) or (int(global_step) % interval != 0):
            return

        max_samples = int(cfg.get("max_samples", 4) or 4)
        max_latents = int(cfg.get("max_latents", 3) or 3)
        max_img_next = int(cfg.get("max_img_next", 16) or 16)
        dump_embeddings = bool(cfg.get("dump_embeddings", False))
        dump_img_next_embeddings = bool(cfg.get("dump_img_next_embeddings", False))
        embeddings_dtype = str(cfg.get("embeddings_dtype", "float16") or "float16").lower()
        embeddings_subdir = str(cfg.get("embeddings_subdir", "embeddings") or "embeddings")
        dump_dir = getattr(self, "_latent_analysis_dump_dir", None)
        if not dump_dir:
            dump_dir = str(cfg.get("dump_dir") or "")
            if not dump_dir:
                out_dir = getattr(self.config, "output_dir", None)
                dump_dir = os.path.join(str(out_dir), "latent_analysis") if out_dir else "latent_analysis"
            try:
                os.makedirs(dump_dir, exist_ok=True)
            except Exception as exc:
                logger.warning(f"[latent_analysis] cannot create dump_dir={dump_dir}: {exc}")
                return
            self._latent_analysis_dump_dir = dump_dir

        # --- locate token positions (do not rely on reasoning_mask which is gated by action-head settings) ---
        input_ids = qwen_inputs.get("input_ids", None)
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return
        attention_mask = qwen_inputs.get("attention_mask", None)
        if attention_mask is None or not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            attention_mask = None

        thinking_token_id = getattr(self.qwen_vl_interface, "thinking_token_id", None)
        img_next_token_id = getattr(self.qwen_vl_interface, "img_next_token_id", None)

        B = int(input_ids.shape[0])
        sample_cap = min(B, max_samples)

        def _base_instruction(text: str) -> str:
            t = (text or "").strip()
            if " @ " in t:
                t = t.split(" @ ", 1)[0].strip()
            return t

        # --- per-sample stats ---
        rows = []
        # Optional: dump embeddings for PCA/UMAP later (store only the needed token vectors).
        think_vec_bank = None
        img_vec_bank = None
        think_mask_bank = None
        img_mask_bank = None

        def _cast_dtype(x: torch.Tensor) -> torch.Tensor:
            if embeddings_dtype in ("fp16", "float16", "half"):
                return x.to(dtype=torch.float16)
            if embeddings_dtype in ("bf16", "bfloat16"):
                return x.to(dtype=torch.bfloat16)
            return x.to(dtype=torch.float32)

        if dump_embeddings:
            H = int(last_hidden.shape[-1])
            think_vec_bank = torch.zeros((sample_cap, max_latents, H), dtype=torch.float16, device="cpu")
            think_mask_bank = torch.zeros((sample_cap, max_latents), dtype=torch.bool, device="cpu")
            if dump_img_next_embeddings and img_next_token_id is not None and max_img_next > 0:
                img_vec_bank = torch.zeros((sample_cap, max_img_next, H), dtype=torch.float16, device="cpu")
                img_mask_bank = torch.zeros((sample_cap, max_img_next), dtype=torch.bool, device="cpu")

        for b in range(sample_cap):
            row = {
                "global_step": int(global_step),
                "batch_index": int(b),
                "use_iterative_forward": bool(use_iterative_forward),
                "cot_mode": str(getattr(self.config.framework, "cot_mode", "implicit")) if self.config is not None else "unknown",
            }

            instr = instructions[b] if b < len(instructions) else ""
            base = _base_instruction(instr)
            row["instruction"] = base[:200]
            row["instruction_sha1"] = hashlib.sha1(base.encode("utf-8")).hexdigest()  # stable task key

            if vlm_outputs is not None:
                try:
                    row["num_reasoning_passes"] = int(vlm_outputs.get("num_reasoning_passes", 0) or 0)
                except Exception:
                    row["num_reasoning_passes"] = None

            # thinking tokens
            think_positions = []
            if thinking_token_id is not None:
                pos = torch.nonzero(input_ids[b] == int(thinking_token_id), as_tuple=False).squeeze(-1)
                if pos.numel() > 0:
                    think_positions = pos[:max_latents].detach().cpu().tolist()
            row["thinking_token_id"] = int(thinking_token_id) if thinking_token_id is not None else None
            row["thinking_positions"] = think_positions
            row["thinking_count"] = int((input_ids[b] == int(thinking_token_id)).sum().item()) if thinking_token_id is not None else 0

            if think_positions:
                vecs = last_hidden[b, torch.tensor(think_positions, device=last_hidden.device), :].detach().float()
                norms = torch.linalg.norm(vecs, dim=-1)
                row["thinking_norm_mean"] = float(norms.mean().item())
                row["thinking_norm_std"] = float(norms.std(unbiased=False).item()) if norms.numel() > 1 else 0.0

                # pairwise cosine among first up-to-3 tokens
                v = F.normalize(vecs, dim=-1)
                cos = (v @ v.T).detach().cpu()
                # store compact off-diagonal mean + specific pairs if available
                if cos.numel() > 1:
                    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
                    row["thinking_cos_offdiag_mean"] = float(off.mean().item()) if off.numel() else None
                else:
                    row["thinking_cos_offdiag_mean"] = None
                if cos.shape[0] >= 2:
                    row["thinking_cos_01"] = float(cos[0, 1].item())
                if cos.shape[0] >= 3:
                    row["thinking_cos_12"] = float(cos[1, 2].item())
                    row["thinking_cos_02"] = float(cos[0, 2].item())

                if dump_embeddings and think_vec_bank is not None and think_mask_bank is not None:
                    token_count = min(len(think_positions), max_latents)
                    think_vec_bank[b, :token_count, :] = _cast_dtype(vecs[:token_count]).cpu()
                    think_mask_bank[b, :token_count] = True

            # img_next tokens (optional, for alignment sanity)
            img_positions = []
            if img_next_token_id is not None:
                pos = torch.nonzero(input_ids[b] == int(img_next_token_id), as_tuple=False).squeeze(-1)
                if pos.numel() > 0:
                    img_positions = pos[:max_img_next].detach().cpu().tolist()
            row["img_next_token_id"] = int(img_next_token_id) if img_next_token_id is not None else None
            row["img_next_count"] = int((input_ids[b] == int(img_next_token_id)).sum().item()) if img_next_token_id is not None else 0
            row["img_next_positions_head"] = img_positions

            if dump_embeddings and dump_img_next_embeddings and img_vec_bank is not None and img_mask_bank is not None and img_positions:
                vecs = last_hidden[b, torch.tensor(img_positions, device=last_hidden.device), :].detach().float()
                token_count = min(len(img_positions), max_img_next)
                img_vec_bank[b, :token_count, :] = _cast_dtype(vecs[:token_count]).cpu()
                img_mask_bank[b, :token_count] = True

            rows.append(row)

        # Append to jsonl
        out_path = os.path.join(str(dump_dir), "latent_stats.jsonl")
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"[latent_analysis] failed to write stats: {exc}")

        # Optional: dump token embeddings for later PCA/UMAP (one file per trigger).
        if dump_embeddings and think_vec_bank is not None and think_mask_bank is not None:
            emb_dir = os.path.join(str(dump_dir), embeddings_subdir)
            try:
                os.makedirs(emb_dir, exist_ok=True)
            except Exception as exc:
                logger.warning(f"[latent_analysis] failed to create embeddings dir: {exc}")
                return

            payload = {
                "global_step": int(global_step),
                "use_iterative_forward": bool(use_iterative_forward),
                "cot_mode": str(getattr(self.config.framework, "cot_mode", "implicit")) if self.config is not None else "unknown",
                "rows": rows,  # small meta, includes instruction_sha1 and positions
                "thinking_vecs": think_vec_bank,          # [N, K, H]
                "thinking_mask": think_mask_bank,         # [N, K]
            }
            if dump_img_next_embeddings and img_vec_bank is not None and img_mask_bank is not None:
                payload["img_next_vecs"] = img_vec_bank   # [N, M, H]
                payload["img_next_mask"] = img_mask_bank  # [N, M]

            # Keep input_ids/attention_mask for exact alignment reproduction in offline analysis.
            try:
                payload["input_ids"] = input_ids[:sample_cap].detach().cpu()
                if attention_mask is not None:
                    payload["attention_mask"] = attention_mask[:sample_cap].detach().cpu()
            except Exception:
                pass

            emb_path = os.path.join(emb_dir, f"latent_emb_step_{int(global_step):08d}.pt")
            try:
                torch.save(payload, emb_path)
            except Exception as exc:
                logger.warning(f"[latent_analysis] failed to write embeddings: {exc}")

    def _compute_img_next_loss(
        self,
        last_hidden: torch.Tensor,
        image_next: List,
        img_next_mask: torch.Tensor,
        fallback_mask: torch.Tensor,
        target_res: int = 112,
    ) -> Optional[torch.Tensor]:
        """
        Compute L1 loss between img_next token hidden states and visual encoder features of next frame.
        """
        if last_hidden is None or image_next is None or len(image_next) == 0:
            return None

        # shape check for mask
        if img_next_mask is None or not torch.any(img_next_mask):
            return None

        device = last_hidden.device
        dtype = last_hidden.dtype

        # Extract predicted embeddings at img_next positions
        try:
            # mask shape [B, L]; expect count per sample = img_next_count (16)
            B = last_hidden.shape[0]
            img_next_count = img_next_mask.sum(dim=1).max().item()
            pred = last_hidden[img_next_mask].view(B, img_next_count, -1)
        except Exception as e:
            logger.warning(f"[img_next_loss] mask reshape failed: {e}")
            return None

        # Encode next images with Qwen3 visual encoder（先手动降采样到 target_res，再交给 processor 归一化）
        try:
            # 获取 processor
            proc = getattr(self.qwen_vl_interface, "processor", None)
            if proc is None and hasattr(self.qwen_vl_interface, "model"):
                proc = getattr(self.qwen_vl_interface.model, "processor", None)
            
            if proc is None:
                logger.warning("[img_next_loss] processor is None, skip img_next_loss")
                return None
            
            # Use only the primary (first) view for img_next loss to match the single-view Bridge setup,
            # while remaining compatible with single-view datasets (non-list entries).
            flat_images = []
            for sample_imgs in image_next:
                if isinstance(sample_imgs, list):
                    flat_images.append(sample_imgs[0] if len(sample_imgs) > 0 else None)
                else:
                    flat_images.append(sample_imgs)
            
            if len(flat_images) == 0:
                logger.warning("[img_next_loss] no images to process")
                return None

            # Resize next-frame images before processor to ensure `res` takes effect.
            if target_res is not None and int(target_res) > 0:
                resized = []
                for img in flat_images:
                    try:
                        resized.append(img.resize((int(target_res), int(target_res))))
                    except Exception:
                        resized.append(img)
                flat_images = resized
            
            # 使用 processor.image_processor 预处理，保持原视觉配置；后续再池化到 16 tokens
            img_processor = getattr(proc, "image_processor", None)
            if img_processor is None:
                logger.warning("[img_next_loss] processor.image_processor is None, skip img_next_loss")
                return None
            with torch.no_grad():
                proc_out = img_processor(images=flat_images, return_tensors="pt")
                proc_out = dict(proc_out)
                pixel_values = proc_out.get("pixel_values", None)
                image_grid_thw = proc_out.get("image_grid_thw", None)
                if pixel_values is None:
                    logger.warning("[img_next_loss] processor returned None pixel_values")
                    return None
                pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)

                main_model = getattr(self.qwen_vl_interface, "model", None)
                if main_model is None:
                    logger.warning("[img_next_loss] main_model.get_image_features not available")
                    return None
                
                # Prefer EMA teacher vision encoder when available.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if hasattr(self.qwen_vl_interface, "get_image_features_target"):
                        img_embeds, _ = self.qwen_vl_interface.get_image_features_target(
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                    else:
                        if not hasattr(main_model, "get_image_features"):
                            logger.warning("[img_next_loss] main_model.get_image_features not available")
                            return None
                        img_embeds, _ = main_model.get_image_features(
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                
                # img_embeds 是 list, 每个元素是 [num_tokens, hidden_dim]
                # 需要 stack 成 [B, num_tokens, hidden_dim]
                if isinstance(img_embeds, (list, tuple)):
                    feats = torch.stack([emb for emb in img_embeds], dim=0).to(device, dtype)
                else:
                    feats = img_embeds.to(device, dtype)
                
                if feats is None or feats.numel() == 0:
                    logger.warning("[img_next_loss] extracted features are empty")
                    return None
                
                # 确保 feats 是 [B, num_tokens, C]
                if feats.dim() == 2:
                    feats = feats.unsqueeze(0)

                # 只保留 2D 池化到 4x4（16 tokens），不做 1D 回退
                grid_side = int(feats.shape[1] ** 0.5)
                target_side = int(img_next_count ** 0.5)
                if grid_side * grid_side != feats.shape[1] or target_side * target_side != img_next_count:
                    logger.warning(f"[img_next_loss] unexpected token grid: tokens={feats.shape[1]}, target={img_next_count}")
                    return None

                feats_2d = feats.transpose(1, 2).reshape(feats.shape[0], feats.shape[2], grid_side, grid_side)
                feats_2d = F.adaptive_avg_pool2d(feats_2d, output_size=(target_side, target_side))
                target_feats = feats_2d.flatten(2).transpose(1, 2)  # [B, target_tokens, C]
        except Exception as e:
            logger.warning(f"[img_next_loss] visual encoding failed: {e}")
            return None

        # Apply fallback mask: skip samples without true next frame
        valid_mask = (~fallback_mask).float().view(-1, 1, 1)
        if valid_mask.sum() <= 0:
            return None

        l1 = torch.nn.functional.l1_loss(pred, target_feats, reduction="none")  # [B, tokens, C]
        mask_full = valid_mask.expand_as(l1)  # broadcast到 token 和通道
        l1 = (l1 * mask_full).sum() / mask_full.sum()
        return l1

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        use_iterative_forward: bool = False,  # ECOT: Enable forward_latent for implicit reasoning
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
             - If use_iterative_forward=True: Use forward_latent for implicit reasoning (ECOT)
             - Otherwise: Use normal forward pass (Baseline)
          3. Action model prediction from hidden states
          4. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            state: Optional proprioceptive state.
            use_iterative_forward: If True, use forward_latent for ECOT implicit reasoning.
                                   This enables multi-pass forward with thinking token embeddings.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], predicted normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # 推理模式与超参
        cot_mode = kwargs.get("cot_mode", "implicit")
        emit_thinking_tokens = kwargs.get("emit_thinking_tokens", False)
        think_max_len = kwargs.get("think_max_len", 64)
        think_temp = kwargs.get("think_temp", 0.1)
        think_topp = kwargs.get("think_topp", 0.9)

        # 显式模式关闭迭代；隐式依赖输入开关
        if cot_mode == "explicit":
            use_iterative_forward = False
        elif cot_mode == "implicit":
            use_iterative_forward = use_iterative_forward
        else:
            use_iterative_forward = False
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        reasoning_mask = self._extract_reasoning_mask(qwen_inputs) if cot_mode == "implicit" else None
        img_next_token_id = getattr(self.qwen_vl_interface, "img_next_token_id", None)
        img_next_mask = (qwen_inputs["input_ids"] == img_next_token_id) if img_next_token_id is not None else None
        
        # Step 2: Choose forward method based on use_iterative_forward flag
        thinking_gen_time = 0.0
        if cot_mode == "explicit":
            # 显式：先生成思维文本，再全量 forward 取 hidden
            gen_result = self.qwen_vl_interface.generate_thinking_explicit(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                max_thinking_len=256,
                temperature=think_temp,
                top_p=think_topp,
            )
            thinking_texts = gen_result.get("thinking_text", [])
            if thinking_texts:
                print(f"[explicit thinking] sample0: {thinking_texts[0]}")
            thinking_gen_time = gen_result["gen_time"]
            full_ids = gen_result["generated_ids"]
            full_mask = torch.ones_like(full_ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    pixel_values=qwen_inputs.get("pixel_values"),
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            reasoning_mask = None

        elif use_iterative_forward and hasattr(self.qwen_vl_interface, 'forward_latent'):
            # ECOT mode: Use forward_latent for implicit reasoning with thinking tokens
            # This performs multiple forward passes with KV-Cache and dynamic embedding updates
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vlm_outputs = self.qwen_vl_interface.forward_latent(
                    input_ids=qwen_inputs["input_ids"],
                    attention_mask=qwen_inputs["attention_mask"],
                    pixel_values=qwen_inputs.get("pixel_values"),
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                )
                # forward_latent returns a dict with 'hidden_states', 'num_reasoning_passes', etc.
                last_hidden = vlm_outputs['hidden_states']  # [B, L, H]
                
                # Optional: Log reasoning passes for debugging
                num_passes = vlm_outputs.get('num_reasoning_passes', 0)
                if num_passes > 0:
                    logger.info(f"[ECOT] Completed {num_passes} reasoning passes in predict_action")
        else:
            # Baseline mode: Normal forward pass (no iterative reasoning)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # last_hidden_state: [B, seq_len, H]
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                reasoning_mask=reasoning_mask,
                img_next_mask=img_next_mask,
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "thinking_gen_time": thinking_gen_time}

    def _extract_reasoning_mask(self, qwen_inputs) -> Optional[torch.Tensor]:
        if not (self.use_reasoning_summary or self.use_reasoning_film):
            return None
        thinking_token_id = getattr(self.qwen_vl_interface, "thinking_token_id", None)
        if thinking_token_id is None:
            return None
        input_ids = qwen_inputs.get("input_ids", None)
        if input_ids is None:
            return None
        mask = (input_ids == thinking_token_id)
        if not torch.any(mask):
            return None
        return mask

    @staticmethod
    def _repeat_reasoning_mask(mask: Optional[torch.Tensor], repeat_steps: int) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if repeat_steps <= 1:
            return mask
        return mask.repeat(repeat_steps, 1)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
