# PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection
## **Accpeted at ICCV 2025** [paper](https://arxiv.org/abs/2507.08979v1)

**PRISM** (Projection-based Reduction of Implicit Spurious bias in vision-language Models) is a data‑free, task‑agnostic framework for mitigating spurious correlations in Vision-Language Models (VLMs) such as CLIP. PRISM leverages Large Language Models (LLMs) to dynamically identify biases and then learns an embedding projection that removes them while preserving semantic alignment.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
4. [Usage](#usage)

   * [PRISM (LLM-Guided Debiasing)](#prism-llm-guided-debiasing)
   * [PRISM-mini (Orthogonal Projection)](#prism-mini-orthogonal-projection)
5. [Results](#results)
6. [Citation](#citation)
7. [License](#license)

---

## Overview

Large-scale pretraining of VLMs often introduces spurious correlations—e.g., associating `camel` with `desert`—which can degrade robustness on underrepresented subpopulations. PRISM addresses this by:

1. **Bias Discovery:** Prompting an LLM (e.g., GPT-4o) to generate scene descriptions that expose spurious label–attribute correlations.
2. **Embedding Projection:** Learning a linear projection via a novel **Latent space Debiasing Loss (LD)** that enforces:

   * **Intra-class invariance:** Align embeddings of the same class across different spurious attributes.
   * **Inter-class separation:** Separate embeddings of different classes sharing the same attribute.

A lightweight variant, **PRISM-mini**, bypasses optimization by computing a closed-form orthogonal projection against identified bias directions.

## Key Features

* **Data-Free:** No external images or bias annotations required for debiasing.
* **Task-Agnostic:** Automatically discovers bias categories from class labels.
* **LLM-Guided:** Utilizes the co-occurrence statistics in LLMs to uncover spurious attributes.
* **Minimal Overhead:** PRISM-mini offers a single-step orthogonal projection for resource-constrained settings.
* **State-of-the-Art:** Achieves top worst-group accuracy (WG) on Waterbirds and CelebA benchmarks while maintaining zero-shot performance.

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch 
* torchvision
* clip @ git+https://github.com/openai/CLIP.git
* wilds


### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MahdiyarMM/PRISM.git
```


## Usage

All experiments assume a CLIP backbone (default: `ViT-L/14`). You can swap to `RN50` via `--model RN50`.

### PRISM (LLM-Guided Debiasing)

1. **Generate scene descriptions** via your chosen LLM (e.g., GPT-4o).
2. **Train the projection** with Latent space Debiasing Loss:

```bash
python main.py \
  --mitigation train \
  --CLIP_model ViT-L/14@336px \
  --dataset waterbirds \
  --batch_size 64 \
  --lr 0.1 \
  --num_samples 500 \
  --epochs 1 \
  --seed 42 \
  --wandb waterbirds_PRISM \
  --init_weight random \
  --num_bases 0 \
  --reg_type None \
  --reg_lambda 1e-3
```


### PRISM-mini (Orthogonal Projection)

1. **Identify spurious attributes** with an LLM:

2. **Apply closed-form projection** at inference:

```bash
python main.py \
  --mitigation orth \
  --dataset celeba \
  --model ViT-L/14
```

This variant requires no further optimization.

## Results

| Method                        | Waterbirds WG ↑ | Acc ↑     | CelebA WG ↑ | Acc ↑     |
| ----------------------------- | --------------- | --------- | ----------- | --------- |
| Zero-shot CLIP                | 36.4%           | 89.3%     | 52.9%       | 72.8%     |
| Orth-Proj \[Chuang et al.]    | 45.3%           | 86.4%     | 41.1%       | 71.1%     |
| VisualDistiller \[Dai et al.] | 42.7%           | 90.6%     | —           | —         |
| **PRISM-mini (ours)**         | 69.5%           | 92.6%     | 82.6%       | 84.4%     |
| **PRISM (ours)**              | **84.2%**       | **93.6%** | **84.0%**   | **86.9%** |

For full comparisons and ablations (LLM choice, number of descriptions, margin sensitivity), see the [paper](https://arxiv.org/abs/2507.08979v1).

## Citation

If you find PRISM useful, please cite our ICCV 2025 paper:

```bibtex
@misc{molahasani2025prism,
      title={PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection}, 
      author={Mahdiyar Molahasani and Azadeh Motamedi and Michael Greenspan and Il-Min Kim and Ali Etemad},
      year={2025},
      eprint={2507.08979},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.08979}, 
}
```

## License

This project is released under the **MIT License**.

