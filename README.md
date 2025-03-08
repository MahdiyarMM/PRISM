# PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection

This repository contains the code for **PRISM (Projection-based Reduction of Implicit Spurious bias in vision-language Models)**, a novel **data-free** and **task-agnostic** debiasing framework for Vision-Language Models (VLMs) like CLIP.

## Overview
PRISM mitigates spurious correlations in CLIP’s embedding space by leveraging Large Language Models (LLMs) to **identify** and **remove** biases in a systematic way. PRISM operates in two main stages:

1. **Bias Discovery:** An LLM is used to generate **scene descriptions** that highlight spurious correlations in CLIP’s text and image embeddings.
2. **Embedding Projection:** A novel contrastive-style **Latent space Debiasing Loss (LD)** is used to learn an optimal transformation that **removes spurious correlations** while preserving semantic alignment between image and text embeddings.

We also provide **PRISM-mini**, an efficient variant that removes biases using a simple **orthogonal projection**, eliminating the need for optimization.

## Features
- **Data-Free:** Does not require external datasets for debiasing.
- **Task-Agnostic:** Works without predefined bias categories.
- **LLM-Guided Bias Identification:** Dynamically identifies spurious correlations using LLMs.
- **Novel Contrastive Debiasing Loss:** Ensures feature disentanglement in embedding space.


## Usage
### 1. PRISM (LLM-Guided Debiasing)
Use the following argument
```bash
$ --mitigation train
```

### 2. PRISM-mini (Fast Orthogonal Projection)
Use the following argument
```bash
$ --mitigation orth
```
This version is computationally efficient and does not require optimization.


## Results
PRISM achieves **state-of-the-art** debiasing performance while maintaining **zero-shot classification accuracy**, outperforming prior data-free debiasing methods on Waterbirds and CelebA datasets.

| **Model**                        | **Waterbirds WG ↑** | **Waterbirds Acc ↑** | **ΔWG ↑**  | **ΔAcc ↑** | **CelebA WG ↑** | **CelebA Acc ↑** | **ΔWG ↑** | **ΔAcc ↑** |
|----------------------------------|---------------------|----------------------|------------|------------|----------------|-----------------|------------|------------|
| **VisualDistiller** [Dai et al.] | 42.7%              | 90.6%               | 6.3%       | 1.3%       | -              | -               | -          | -          |
| **Orth-Proj** [Chuang et al.]    | 45.3%              | 86.4%               | 8.9%       | -2.9%      | 71.1%          | **87.0%**       | -1.7%      | **-0.6%**  |
| **Orth-Cali** [Chuang et al.]    | 68.8%              | 84.5%               | 32.4%      | -4.3%      | 76.1%          | 86.2%           | 3.3%       | -1.4%      |
| **RoboShot** [Adila et al.]      | 45.2%              | 79.2%               | 12.8%      | -10.1%     | _82.6%_        | 85.5%           | _9.8%_     | -2.1%      |
| **PRISM-mini (ours)**            | _69.5%_            | _92.6%_             | _33.1%_    | _3.3%_     | _82.6%_        | 84.4%           | _9.8%_     | -3.2%      |
| **PRISM (ours)**                 | **84.2%**          | **93.6%**           | **47.8%**  | **4.3%**   | **84.0%**      | _86.9%_         | **11.2%**  | _-0.7%_    |



## License
MIT License.

