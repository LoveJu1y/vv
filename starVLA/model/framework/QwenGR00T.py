# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
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
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format (tokenization and thinking token alignment if enabled)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions
        )
        reasoning_mask = (
            self._extract_reasoning_mask(qwen_inputs)
            if self.training_stage != "reasoning_only"
            else None
        )
        
        # Check if iterative implicit reasoning is enabled
        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        use_iterative_forward = enable_latent_reasoning and hasattr(self.qwen_vl_interface, 'forward_latent')
        
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

        # Step 3: Compute losses based on training stage
        result = {}
        
        if self.training_stage == "reasoning_only":
            # Stage 1: Only train VLM reasoning, skip action head
            if vlm_loss is None:
                raise ValueError(
                    "training_stage='reasoning_only' requires VLM loss, but vlm_loss is None. "
                    "Please ensure enable_latent_reasoning=True and labels are provided."
                )
            result["vlm_loss"] = vlm_loss
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
                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                    reasoning_mask=reasoning_mask_repeated,
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
                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                    reasoning_mask=reasoning_mask_repeated,
                )

            result["action_loss"] = action_loss
            
            # Combine with VLM loss if available
        if vlm_loss is not None:
            vlm_loss_weight = self.config.framework.get("latent_reasoning", {}).get("vlm_loss_weight", 0.5)
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = action_loss + vlm_loss_weight * vlm_loss
        else:
            result["total_loss"] = action_loss

        return result

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
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        reasoning_mask = self._extract_reasoning_mask(qwen_inputs)
        
        # Step 2: Choose forward method based on use_iterative_forward flag
        if use_iterative_forward and hasattr(self.qwen_vl_interface, 'forward_latent'):
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
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

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
