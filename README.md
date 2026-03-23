# StarVLA

StarVLA is a modular codebase for building vision-language-action models. It
keeps model, data, training, evaluation, and deployment boundaries explicit so
new VLA ideas can be implemented and debugged quickly.

The current maintained open-source mainline in this repository is:

- `QwenGR00T` as the primary framework
- implicit latent reasoning with iterative forward passes
- `starVLA/training/train.py` as the training entrypoint
- Bridge and LIBERO as the main training / evaluation workflows

This README intentionally prioritizes the current recommended path instead of
trying to document every historical experiment at once.

![](assets/Framworks.png)

## Core Idea

In the implicit reasoning path, the model does not have to emit a long explicit
chain-of-thought string. Instead, reasoning slots are represented by thinking
tokens, and the VLM performs iterative forward passes:

1. run forward until the next thinking token
2. take the hidden state before that token
3. feed that hidden state back as the next reasoning token embedding
4. continue the next forward pass

The final hidden states are then consumed by the action head for policy
prediction.

This logic is centered around:

- `starVLA/model/framework/QwenGR00T.py`
- `starVLA/model/modules/vlm/QWen3.py`
- `starVLA/training/train.py`

## Highlights

- Multiple VLA framework implementations in one codebase
- Implicit latent reasoning with iterative forward and hidden-state feedback
- Modular dataloaders and framework boundaries
- Training, evaluation, and deployment scripts for Bridge / LIBERO / SimplerEnv
- Support for both action-token and continuous-action policy heads

## Current Focus

If you are new to this repository, start with the following path:

- Training entrypoint: `starVLA/training/train.py`
- Single-run training scripts:
  - `scripts/run_starvla_bridge.sh`
  - `scripts/run_starvla_libero.sh`
- Multi-stage training scripts:
  - `scripts/run_bridge_multistage.sh`
  - `scripts/run_libero_multistage.sh`
- LIBERO evaluation:
  - `examples/LIBERO/eval_libero_all.sh`
- Policy server deployment:
  - `deployment/model_server/server_policy.py`

## Repository Layout

- `starVLA/`: model, data, training, configs
- `scripts/`: recommended training launchers
- `examples/LIBERO/`: LIBERO evaluation entrypoints
- `examples/SimplerEnv/`: SimplerEnv evaluation entrypoints
- `deployment/`: policy server and deployment helpers
- `docs/`: cleanup plans and release planning

## Installation

```bash
git clone https://github.com/starVLA/starVLA
cd starVLA

conda create -n starVLA python=3.10 -y
conda activate starVLA

pip install -r requirements.txt
pip install -e .
```

If you want to use `flash_attention_2`, install a compatible `flash-attn`
version for your CUDA and PyTorch environment. Some configs in this repo use
`sdpa`, so FlashAttention is not always required.

Useful environment checks:

```bash
nvcc -V
python -c "import torch; print(torch.__version__)"
pip list | grep -E 'torch|transformers|flash-attn'
```

## Quick Start

### 1. Import smoke test

```bash
python -c "from starVLA.training.train import main; print('OK')"
```

### 2. Config syntax check

```bash
python - <<'PY'
from omegaconf import OmegaConf
for path in [
    "starVLA/config/training/bridge_lerobot_stage2.yaml",
    "starVLA/config/training/libero_all_ecot_stage4.yaml",
    "starVLA/config/training/libero_goal_ecot_stage4.yaml",
]:
    OmegaConf.load(path)
    print("OK:", path)
PY
```

### 3. Lint check

```bash
make check
```

## Training

The canonical training entrypoint is:

```bash
starVLA/training/train.py
```

The repository currently ships three main training configs:

- `starVLA/config/training/bridge_lerobot_stage2.yaml`
- `starVLA/config/training/libero_all_ecot_stage4.yaml`
- `starVLA/config/training/libero_goal_ecot_stage4.yaml`

### Bridge training

Single-run launcher:

```bash
bash scripts/run_starvla_bridge.sh
```

Multi-stage launcher:

```bash
bash scripts/run_bridge_multistage.sh
```

### LIBERO training

Single-run launcher:

```bash
bash scripts/run_starvla_libero.sh
```

Multi-stage launcher:

```bash
bash scripts/run_libero_multistage.sh
```

### Manual training example

```bash
torchrun \
  --nproc_per_node=8 \
  --master_port=29512 \
  starVLA/training/train.py \
  --config_yaml starVLA/config/training/bridge_lerobot_stage2.yaml \
  --run_root_dir results/Bridge \
  --run_id bridge_debug
```

### Training notes

- The current maintained path is implicit latent reasoning
- Most practical runs are launched through the scripts in `scripts/`
- Many config values are intended to be overridden from the command line
- `run_root_dir` stores config snapshots and training outputs

## Evaluation

### LIBERO

Parallel evaluation:

```bash
bash examples/LIBERO/eval_libero_all.sh /abs/path/to/checkpoint.pt
```

The evaluation script starts policy servers and then launches LIBERO workers
against them. Check the script header for environment variables such as Python
interpreters, GPU allocation, and output directories.

### SimplerEnv

See:

- `examples/SimplerEnv/README.md`
- `examples/SimplerEnv/bridge_eval.sh`

If you are validating an existing checkpoint, SimplerEnv is usually the fastest
end-to-end policy sanity check after deployment is set up.

## Deployment

Start a policy server from a trained checkpoint:

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path /abs/path/to/checkpoint.pt \
  --port 10093 \
  --use_bf16
```

The deployment stack is organized under:

- `deployment/model_server/server_policy.py`
- `deployment/model_server/tools/websocket_policy_server.py`
- `deployment/model_server/tools/websocket_policy_client.py`

For local protocol debugging, see:

- `deployment/model_server/debug_server_policy.py`

## Model Zoo

Available public checkpoints include:

| Model | Description | Link |
| --- | --- | --- |
| `Qwen2.5-VL-3B-Action` | Qwen2.5-VL with action tokens | [Hugging Face](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action) |
| `Qwen3-VL-4B-Action` | Qwen3-VL with action tokens | [Hugging Face](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action) |
| `QWen-GR00T-Bridge` | QwenVL + GR00T action head | [Hugging Face](https://huggingface.co/StarVLA/Qwen-GR00T-Bridge) |
| `QWen3VL-GR00T-Bridge-RT-1` | Qwen3VL + GR00T action head | [Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1) |

Additional checkpoints are listed on the StarVLA Hugging Face page.

## Design Principles

StarVLA tries to keep VLA research iteration fast through a few simple rules:

- Dataloaders return raw, model-agnostic samples
- Frameworks own model-specific preprocessing
- `forward()` and `predict_action()` stay close to raw task inputs
- Configs are centralized and easy to override from CLI
- Submodules can be debugged independently

## FAQ

### Why not put preprocessing inside the dataloader?

Keeping preprocessing inside the framework allows model-specific handling while
keeping the dataloader generic and reusable.

### Can I swap the VLM backbone?

Yes. The codebase is structured so new vision-language modules can be wired into
the framework layer without rewriting the whole training pipeline.

### Can I override parameters from the command line?

Yes. The repo uses OmegaConf and expects config overrides from the CLI.

### Can I freeze modules or use different learning rates?

Yes. Freeze behavior and module-specific learning rates are controlled from the
trainer config and CLI overrides.

### Can I resume from a checkpoint?

Yes. Use `trainer.pretrained_checkpoint` and, if needed,
`trainer.reload_modules`. The current training pipeline focuses on model-state
loading rather than full optimizer-state resumption.

## Limitations

- The open-source mainline currently prioritizes implicit latent reasoning
- Some scripts still assume local dataset preparation outside this repository
- Dataset annotations and caches may need project-specific preparation
- Documentation is being updated to match the cleaned code path

## Citation

```bibtex
@misc{starvla2025,
  title  = {StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing},
  author = {StarVLA Community},
  url    = {https://github.com/starVLA/starVLA},
  year   = {2025}
}
```

## License

StarVLA is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

- Open an issue first for bugs, regressions, or design discussions
- Run `make check` before submitting a PR
- Keep PRs scoped and explain any config or behavior changes clearly

More project-process documents will be added as the repository cleanup
continues.

## Acknowledgements

This repository draws inspiration from and builds upon several open-source
projects, including:

- [LeRobot](https://github.com/huggingface/lerobot)
- [GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Qwen-VL](https://github.com/QwenLM/Qwen3-VL/tree/main)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1)
