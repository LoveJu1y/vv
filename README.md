<h1 align="center">LaRA-VLA</h1>

<p align="center">
  <strong>Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models</strong>
</p>

<p align="center">
  Shuanghao Bai · Jing Lyu · Wanqi Zhou · Zhe Li · Dakai Wang · Lei Xing ·
  Xiaoguang Zhao · Pengwei Wang · Zhongyuan Wang · Cheng Chi · Badong Chen ·
  Shanghang Zhang
</p>



<p align="center">
  <a href="https://loveju1y.github.io/Latent-Reasoning-VLA/">
    <img src="https://img.shields.io/badge/Homepage-LaRA--VLA-2d6cdf?style=for-the-badge" alt="Homepage">
  </a>
  <a href="https://arxiv.org/abs/2602.01166">
    <img src="https://img.shields.io/badge/arXiv-2602.01166-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
  <a href="https://github.com/LoveJu1y/LaRA-VLA">
    <img src="https://img.shields.io/badge/GitHub-LaRA--VLA-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  <img src="assets/2.png" alt="LaRA-VLA overview" width="92%">
</p>

<p align="center">
  <sub>
    LaRA-VLA performs iterative latent reasoning by feeding hidden states back into reasoning slots
    before action prediction, rather than relying on long explicit chain-of-thought generation.
  </sub>
</p>

## Method Overview

This repository builds on the open-source StarVLA codebase and focuses on
**implicit latent reasoning** for VLA policy learning. The active code
namespace in this repository is `laravla/`.

## Results

### LIBERO

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <thead>
    <tr style="border-bottom: 2px solid black;">
      <th>CoT Type</th>
      <th>Method</th>
      <th>Spatial</th>
      <th>Goal</th>
      <th>Object</th>
      <th>Long</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <!-- No CoT -->
    <tr>
      <td rowspan="3"><b>No CoT</b></td>
      <td>OpenVLA (Kim et al., 2025b)</td>
      <td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td>
    </tr>
    <tr>
      <td>π₀ (Black et al., 2024)</td>
      <td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td>
    </tr>
    <tr style="border-bottom: 2px solid black;">
      <td>OpenVLA-OFT (Kim et al., 2025a)</td>
      <td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td>
    </tr>

    <!-- Textual CoT -->
    <tr>
      <td rowspan="4"><b>Textual CoT</b></td>
      <td>ThinkAct (Huang et al., 2025)</td>
      <td>88.3</td><td>91.4</td><td>87.1</td><td>70.9</td><td>84.4</td>
    </tr>
    <tr>
      <td>MolmoAct (Lee et al., 2025)</td>
      <td>87.0</td><td>95.4</td><td>87.6</td><td>77.2</td><td>86.6</td>
    </tr>
    <tr>
      <td>π₀.₅ (Intelligence et al., 2025)</td>
      <td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.8</td>
    </tr>
    <tr style="border-bottom: 2px solid black;">
      <td>DeepThinkVLA (Yin et al., 2025)</td>
      <td>99.0</td><td>96.6</td><td>96.4</td><td>96.2</td><td>97.0</td>
    </tr>

    <!-- Visual CoT -->
    <tr>
      <td rowspan="4"><b>Visual CoT</b></td>
      <td>CoT-VLA (Zhao et al., 2025)</td>
      <td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>81.1</td>
    </tr>
    <tr>
      <td>DreamVLA (Zhang et al., 2025b)</td>
      <td>97.5</td><td>94.0</td><td>89.5</td><td>89.5</td><td>92.6</td>
    </tr>
    <tr>
      <td>F1 (Lv et al., 2025)</td>
      <td>98.2</td><td>97.8</td><td>95.4</td><td>91.3</td><td>95.7</td>
    </tr>
    <tr style="border-bottom: 2px solid black;">
      <td>UD-VLA (Chen et al., 2025b)</td>
      <td>94.1</td><td>95.7</td><td>91.2</td><td>89.6</td><td>92.7</td>
    </tr>

    <!-- Latent CoT -->
    <tr>
      <td rowspan="2"><b>Latent CoT</b></td>
      <td>Fast-ThinkAct (Huang et al., 2026)</td>
      <td>92.0</td><td>97.2</td><td>90.2</td><td>79.4</td><td>89.7</td>
    </tr>
    <tr>
      <td><b>LaRA-VLA (Ours)</b></td>
      <td>96.4</td><td>98.6</td><td>99.8</td><td>96.6</td><td><b>97.9</b></td>
    </tr>
  </tbody>
</table>

### Bridge

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <thead>
    <tr style="border-bottom: 2px solid black;">
      <th>CoT Type</th>
      <th>Method</th>
      <th>Put Spoon</th>
      <th>Put Carrot</th>
      <th>Stack Block</th>
      <th>Put Eggplant</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <!-- No CoT -->
    <tr>
      <td rowspan="5"><b>No CoT</b></td>
      <td>OpenVLA (Kim et al., 2025b)</td>
      <td>0.0</td><td>0.0</td><td>0.0</td><td>4.1</td><td>1.0</td>
    </tr>
    <tr>
      <td>Octo (Ghosh et al., 2024)</td>
      <td>47.2</td><td>9.7</td><td>4.2</td><td>56.9</td><td>29.5</td>
    </tr>
    <tr>
      <td>OpenVLA-OFT (Kim et al., 2025a)</td>
      <td>12.5</td><td>4.2</td><td>8.3</td><td>37.5</td><td>39.6</td>
    </tr>
    <tr>
      <td>π₀ (Black et al., 2024)</td>
      <td>29.1</td><td>0.0</td><td>16.7</td><td>62.5</td><td>40.1</td>
    </tr>
    <tr style="border-bottom: 2px solid black;">
      <td>CogACT (Li et al., 2024)</td>
      <td>71.7</td><td>50.8</td><td>15.0</td><td>67.5</td><td>51.3</td>
    </tr>

    <!-- Textual CoT -->
    <tr style="border-bottom: 2px solid black;">
      <td><b>Textual CoT</b></td>
      <td>ThinkAct (Huang et al., 2025)</td>
      <td>58.3</td><td>37.5</td><td>8.7</td><td>70.8</td><td>43.8</td>
    </tr>

    <!-- Visual CoT -->
    <tr>
      <td rowspan="2"><b>Visual CoT</b></td>
      <td>F1 (Lv et al., 2025)</td>
      <td>50.0</td><td>70.8</td><td>50.0</td><td>66.7</td><td>59.4</td>
    </tr>
    <tr style="border-bottom: 2px solid black;">
      <td>UD-VLA (Chen et al., 2025b)</td>
      <td>58.3</td><td>62.5</td><td>54.1</td><td>75.0</td><td>62.5</td>
    </tr>

    <!-- Latent CoT -->
    <tr>
      <td><b>Latent CoT</b></td>
      <td><b>LaRA-VLA (Ours)</b></td>
      <td>95.8</td><td>62.5</td><td>25.0</td><td>91.7</td><td><b>68.8</b></td>
    </tr>
  </tbody>
</table>

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

The LIBERO results above correspond to the evaluation workflow documented in
[examples/LIBERO/README.md](examples/LIBERO/README.md).

### SimplerEnv

The Bridge real-world results above are evaluated through the SimplerEnv-based
pipeline documented in
[examples/SimplerEnv/README.md](examples/SimplerEnv/README.md).




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
