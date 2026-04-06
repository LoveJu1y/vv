# LaRA-VLA

**Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models**

- Project page: [Latent Reasoning VLA](https://loveju1y.github.io/Latent-Reasoning-VLA/)
- Paper: [arXiv:2602.01166](https://arxiv.org/abs/2602.01166)
- Code: [LoveJu1y/LaRA-VLA](https://github.com/LoveJu1y/LaRA-VLA)

## Release Status

This open-source release currently includes:

- ✅ training code
- ✅ evaluation code
- ⏳ model weights (coming soon)
- ⏳ datasets / processed annotations (coming soon)

## Method Overview

LaRA-VLA is a latent-reasoning VLA project for vision-language-action policy
learning. The public project identity of this repository is **LaRA-VLA**.

The active code namespace in this repository is now `laravla/`.

The current focus of LaRA-VLA is **implicit latent reasoning** for VLA policy
learning.

![](assets/2.png)

## Results

### LIBERO

![](assets/3.png)

### Bridge

![](assets/5.png)

## Installation

```bash
git clone https://github.com/LoveJu1y/LaRA-VLA
cd LaRA-VLA

conda create -n lara-vla python=3.10 -y
conda activate lara-vla

pip install -r requirements.txt
pip install -e .
```


## Quick Start

### 1) Basic check

```bash
python -c "from laravla.training.train import main; print('OK')"
```

### 2) Multi-stage training for VLM 

Bridge:

```bash
bash scripts/run_bridge_multistage.sh
```

LIBERO:

```bash
bash scripts/run_libero_multistage.sh
```

### 3) Single-stage training for VLA

Bridge:

```bash
bash scripts/run_laravla_bridge.sh
```

LIBERO:

```bash
bash scripts/run_laravla_libero.sh
```


## Evaluation

### LIBERO

Please refer to:

- `/examples/LIBERO/README.md`

### SimplerEnv

Please refer to:

- `examples/SimplerEnv/README.md`




## Citation

```bibtex
@article{bai2026latentreasoningvla,
  title={Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models},
  author={Bai, Shuanghao and Lyu, Jing and Zhou, Wanqi and Li, Zhe and Wang, Dakai and Xing, Lei and Zhao, Xiaoguang and Wang, Pengwei and Wang, Zhongyuan and Chi, Cheng and Chen, Badong and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2602.01166},
  year={2026}
}
```

## License

Released under the MIT License. See `LICENSE`.
