# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT



from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    use_reasoning_summary: bool = field(
        default=False, metadata={"help": "Enable reasoning latent summarizer for conditioning."}
    )
    reasoning_summary_tokens: int = field(
        default=2, metadata={"help": "Number of summary tokens concatenated to future tokens."}
    )
    reasoning_summary_heads: int = field(
        default=4, metadata={"help": "Attention heads inside the reasoning summarizer."}
    )
    reasoning_summary_dropout: float = field(
        default=0.1, metadata={"help": "Dropout ratio for the reasoning summarizer."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class ReasoningSummarizer(nn.Module):
    """
    Summarize reasoning (<|thinking|>) latents into a small set of tokens for FiLM/DiT.
    """

    def __init__(
        self,
        vlm_hidden_dim: int,
        target_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Two fixed queries: one偏向scale，一偏向shift
        self.num_summary_tokens = 2
        self.input_proj = nn.Linear(vlm_hidden_dim, target_dim)
        self.latent_norm = nn.LayerNorm(target_dim)
        self.summary_queries = nn.Parameter(torch.randn(self.num_summary_tokens, target_dim))
        nn.init.normal_(self.summary_queries, mean=0.0, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, target_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(target_dim * 4, target_dim),
        )

    def forward(self, vl_embs: torch.Tensor, latent_mask: torch.Tensor) -> Optional[torch.Tensor]:
        if latent_mask is None:
            return None
        latent_mask = latent_mask.to(device=vl_embs.device)
        if latent_mask.dtype != torch.bool:
            latent_mask = latent_mask.bool()
        max_latents = int(latent_mask.sum(dim=1).max().item())
        if max_latents == 0:
            return None

        latents, valid_mask = self._gather_latents(vl_embs, latent_mask, max_latents)
        latents = self.latent_norm(self.input_proj(latents))

        summary = vl_embs.new_zeros(latents.shape[0], self.num_summary_tokens, latents.shape[-1])
        query = self.summary_queries.unsqueeze(0).expand(latents.shape[0], -1, -1)
        valid_batch = valid_mask.any(dim=1)

        if valid_batch.any():
            lat_valid = latents[valid_batch]
            mask_valid = valid_mask[valid_batch]
            query_valid = query[valid_batch]
            attn_out, _ = self.attn(
                query_valid,
                lat_valid,
                lat_valid,
                key_padding_mask=~mask_valid,
            )
            attn_out = query_valid + self.dropout(attn_out)
            attn_out = attn_out + self.ff(attn_out)
            # Keep dtype aligned with target buffer to avoid bfloat16/float mismatch
            attn_out = attn_out.to(summary.dtype)
            summary[valid_batch] = attn_out
        return summary
    def _gather_latents(
        self,
        vl_embs: torch.Tensor,
        latent_mask: torch.Tensor,
        max_latents: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect latent tokens per-sample, padding to max_latents.
        """
        B, _, hidden_dim = vl_embs.shape
        latents = vl_embs.new_zeros(B, max_latents, hidden_dim)
        valid_mask = latent_mask.new_zeros(B, max_latents)

        # Fast path: assume thinking tokens已对齐（常见于预处理对齐后的批次）
        first_mask = latent_mask[0]
        if first_mask.any():
            aligned = torch.all(latent_mask == first_mask, dim=1).all()
            if aligned:
                idx = torch.nonzero(first_mask, as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    token_count = min(idx.numel(), max_latents)
                    gathered = vl_embs[:, idx[:token_count], :]
                    latents[:, :token_count, :] = gathered
                    valid_mask[:, :token_count] = True
                    return latents, valid_mask

        for b in range(B):
            indices = torch.nonzero(latent_mask[b], as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            token_feats = vl_embs[b, indices, :]
            token_count = min(token_feats.shape[0], max_latents)
            latents[b, :token_count] = token_feats[:token_count]
            valid_mask[b, :token_count] = True
        return latents, valid_mask


class ReasoningFiLM(nn.Module):
    """
    Generate FiLM scale/shift from summary latents.
    """

    def __init__(
        self,
        hidden_dim: int,
        mid_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.scale_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
        )
        self.shift_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
        )
        # 初始化为“无调制”
        nn.init.zeros_(self.scale_mlp[-1].weight)
        nn.init.zeros_(self.scale_mlp[-1].bias)
        nn.init.zeros_(self.shift_mlp[-1].weight)
        nn.init.zeros_(self.shift_mlp[-1].bias)

    def forward(self, summary_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # summary_tokens: [B, 2, H] （0 -> scale, 1 -> shift）
        summary_tokens = self.ln(summary_tokens)
        scale_src = summary_tokens[:, 0]
        shift_src = summary_tokens[:, 1]
        scale = self.scale_mlp(scale_src)
        shift = self.shift_mlp(shift_src)
        return scale, shift

class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size # 是不要和 Q对齐？
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.use_reasoning_summary = getattr(config, "use_reasoning_summary", False)  # 控制是否拼接 summary token
        self.use_reasoning_film = getattr(config, "use_reasoning_film", False)        # 控制是否启用 FiLM 调制
        # 构建 summarizer：只要 FiLM 或 summary token 任一开启就构建
        if self.use_reasoning_summary or self.use_reasoning_film:
            cross_attention_dim = diffusion_model_cfg.get("cross_attention_dim", self.input_embedding_dim)
            self.reasoning_summarizer = ReasoningSummarizer(
                vlm_hidden_dim=cross_attention_dim,
                target_dim=self.input_embedding_dim,
                num_heads=getattr(config, "reasoning_summary_heads", 4),
                dropout=getattr(config, "reasoning_summary_dropout", 0.1),
            )
        else:
            self.reasoning_summarizer = None
        if self.use_reasoning_film:
            self.reasoning_film = ReasoningFiLM(
                hidden_dim=self.input_embedding_dim,
                mid_dim=getattr(config, "reasoning_film_hidden", self.input_embedding_dim),
                dropout=getattr(config, "reasoning_film_dropout", 0.1),
            )
            self.film_first_k = max(0, int(getattr(config, "reasoning_film_first_k", 0)))
        else:
            self.reasoning_film = None
            self.film_first_k = 0

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _build_reasoning_summary_tokens(
        self,
        vl_embs: torch.Tensor,
        reasoning_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_reasoning_summary or self.reasoning_summarizer is None:
            return None
        if reasoning_mask is None:
            return None
        if vl_embs.shape[:2] != reasoning_mask.shape:
            return None
        return self.reasoning_summarizer(vl_embs, reasoning_mask)

    def _build_reasoning_modulation(
        self,
        vl_embs: torch.Tensor,
        reasoning_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self.use_reasoning_film
            or self.film_first_k <= 0
            or self.reasoning_summarizer is None
            or self.reasoning_film is None
        ):
            return None, None
        if reasoning_mask is None or vl_embs.shape[:2] != reasoning_mask.shape:
            return None, None
        summary_tokens = self.reasoning_summarizer(vl_embs, reasoning_mask)
        if summary_tokens is None or summary_tokens.shape[1] < 2:
            return None, None
        scale, shift = self.reasoning_film(summary_tokens)
        # 对齐 dtype/device
        scale = scale.to(vl_embs.dtype)
        shift = shift.to(vl_embs.dtype)
        return scale, shift

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        reasoning_mask: Optional[torch.Tensor] = None,
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        
        # embed state
        state_features = self.state_encoder(state) if state is not None else None
        print(666) if state is not None else "state is None"


        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        summary_tokens = self._build_reasoning_summary_tokens(vl_embs, reasoning_mask)
        film_scale, film_shift = self._build_reasoning_modulation(vl_embs, reasoning_mask)
        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        seq_chunks = []
        if state_features is not None:
            seq_chunks.append(state_features)
        if summary_tokens is not None:
            seq_chunks.append(summary_tokens)
        seq_chunks.extend([future_tokens, action_features])
        sa_embs = torch.cat(seq_chunks, dim=1)

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            modulation=(film_scale, film_shift) if film_scale is not None and film_shift is not None else None,
            film_first_k=self.film_first_k,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        reasoning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        state_features = self.state_encoder(state) if state is not None else None

        summary_tokens = self._build_reasoning_summary_tokens(vl_embs, reasoning_mask)
        film_scale, film_shift = self._build_reasoning_modulation(vl_embs, reasoning_mask)
        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            seq_chunks = []
            if state_features is not None:
                seq_chunks.append(state_features)
            if summary_tokens is not None:
                seq_chunks.append(summary_tokens)
            seq_chunks.extend([future_tokens, action_features])
            sa_embs = torch.cat(seq_chunks, dim=1)


            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                modulation=(film_scale, film_shift) if film_scale is not None and film_shift is not None else None,
                film_first_k=self.film_first_k,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(
        full_config=config
    )


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
