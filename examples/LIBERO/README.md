## LIBERO

This directory contains the recommended LIBERO evaluation entrypoints.

Main files:

- `eval_libero.py`: evaluate one LIBERO suite against one policy server
- `eval_libero_all.sh`: recommended parallel multi-suite evaluation
- `run_all_ckpts_libero_all.sh`: batch evaluation for many checkpoints

## Prerequisites

You usually need:

- one trained checkpoint
- one LaRA-VLA Python environment (the code package namespace remains `starVLA`)
- one LIBERO Python environment
- `LIBERO_HOME`

Useful checks:

```bash
python -c "from starVLA.training.train import main; print('OK')"
python -c "from libero.libero import benchmark; print('OK')"
```

## Recommended: Parallel Evaluation

```bash
STAR_VLA_PYTHON=/path/to/starvla/python \
LIBERO_PYTHON=/path/to/libero/python \
LIBERO_HOME=/path/to/LIBERO \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TASK_SUITES=libero_goal,libero_spatial,libero_object,libero_10 \
bash examples/LIBERO/eval_libero_all.sh /abs/path/to/checkpoint.pt
```


Outputs are written under:

```text
<checkpoint_dir>/eval_libero_implicit_parallel/<checkpoint_name>/
```

## Batch Evaluation for Many Checkpoints

```bash
STAR_VLA_PYTHON=/path/to/starvla/python \
LIBERO_PYTHON=/path/to/libero/python \
LIBERO_HOME=/path/to/LIBERO \
bash examples/LIBERO/run_all_ckpts_libero_all.sh /abs/path/to/checkpoints
```


