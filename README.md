# Evolving Sparsity: Leveraging Token Importance Dynamics for Efficient LLM Decoding with Sparse Attention

> **EvoSparse** exploits the evolution of token importance across decoding steps and model layers for efficient sparse attention in long-context LLM inference.

**ACL 2026 Main Conference** · [Paper](https://aclanthology.org/2026.acl-long.530/) · [Code](https://github.com/iLearn-Lab/ACL26-EvoSparse)

## Authors

**Ruizi Han**<sup>1,2</sup>, **Miao Zhang**<sup>1</sup>\*, **Ziyue Qiao**<sup>2</sup>\*, **Liqiang Nie**<sup>1</sup>

<sup>1</sup> Harbin Institute of Technology (Shenzhen)  
<sup>2</sup> Great Bay University  
\* Co-corresponding authors

## Links

- **Paper:** [ACL Anthology](https://aclanthology.org/2026.acl-long.530/)
- **PDF:** [ACL 2026 Paper](https://aclanthology.org/2026.acl-long.530.pdf)
- **Code Repository:** [GitHub](https://github.com/iLearn-Lab/ACL26-EvoSparse)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Method API](#method-api)
  - [PG-19 Perplexity](#pg-19-perplexity)
  - [LongBench](#longbench)
  - [RULER](#ruler)
- [Citation](#citation)
- [License](#license)

---

## Updates

- **07/2026:** EvoSparse was published at ACL 2026 (Main Conference).

## Introduction

This repository provides the **official implementation** of **EvoSparse**, introduced in *Evolving Sparsity: Leveraging Token Importance Dynamics for Efficient LLM Decoding with Sparse Attention* (ACL 2026).

During autoregressive decoding, attention computation grows linearly with context length. EvoSparse models token importance as a dynamic process across decoding steps and model layers, reducing repeated importance estimation while preserving both long-term context and query-relevant information.

The repository includes the shared EvoSparse implementation, model-specific retrieval-head configurations, and evaluation workflows for **PG-19**, **LongBench**, and **RULER**.


## Method / Framework

EvoSparse combines two complementary components to exploit token-importance dynamics during decoding:

1. **Cross-Step Accumulation** maintains decayed token-importance scores across decoding steps, avoiding redundant computation of long-term importance.
2. **Cross-Layer Propagation** identifies query-aware token blocks at retrieval heads and reuses their selected indices across subsequent layers.

Together, they preserve stable context information and adapt to the current query. Further algorithmic details are provided in the [paper](https://aclanthology.org/2026.acl-long.530/).

## Project Structure

The following layout summarizes the components described in the repository's original README:

```text
ACL26-EvoSparse/
├── src/evosparse/             # EvoSparse implementation
├── configs/retrieval_heads/   # Model-specific retrieval-head scores
├── experiments/
│   ├── pg19/                 # PG-19 perplexity evaluation
│   ├── ruler/                # RULER generation, inference, and evaluation
│   └── longbench/            # LongBench inference and evaluation
├── figures/                  # Plotting scripts and generated figures
├── results/                  # Local experiment outputs
├── pyproject.toml            # Project and benchmark dependencies
├── README.md
└── LICENSE
```

## Installation

### Environment

The original experimental environment uses:

- Python **3.11.13**
- PyTorch **2.8.0** with CUDA **12.8**
- Transformers **4.52.4**
- Datasets **3.6.0**

Clone the repository and install EvoSparse with its benchmark dependencies:

```bash
git clone https://github.com/iLearn-Lab/ACL26-EvoSparse.git
cd ACL26-EvoSparse

conda create -n evosparse python=3.11 -y
conda activate evosparse

# Install the CUDA 12.8 build used in our experiments.
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install EvoSparse and the pinned benchmark dependencies from pyproject.toml.
pip install -e ".[experiments]"
```

## Usage

### Method API

The package exposes functions for enabling EvoSparse and resetting its state between independent samples:

```python
from evosparse import enable_evosparse, reset_evosparse_state

enable_evosparse(
    model,
    group_size=16,
    total_budget=4096,
    sink_token=16,
    local_token=128,
    per_head_top_group=15,
    retrieval_head_num=20,
    decay=0.996,
)

# Call this before processing an independent sample with the same patched model.
reset_evosparse_state(model)
```

The package also exports the two ablations used in the paper:

```python
from evosparse import (
    enable_evosparse_without_cross_layer,
    enable_evosparse_without_cross_step,
)
```

### PG-19 Perplexity

Run commands from the **repository root**. The following EvoSparse configuration uses a token budget of 4096, block size 16, 20 retrieval heads, and 15 candidate groups per retrieval head.

**Dense attention:**

```bash
python experiments/pg19/evaluate.py \
  --model_name_or_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --attention dense \
  --num_eval_tokens 32000 \
  --output_dir results/pg19_dense
```

**EvoSparse:**

```bash
python experiments/pg19/evaluate.py \
  --model_name_or_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --attention evosparse \
  --budget 4096 \
  --block_size 16 \
  --retrieval_heads 20 \
  --groups_per_head 15 \
  --num_eval_tokens 32000 \
  --output_dir results/pg19_evosparse_b4096_bs16_h20_g15
```

### LongBench

**Inference:**

```bash
cd experiments/longbench

python pred.py \
  --model gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --task qasper \
  --attn_type evosparse \
  --budget 4096 \
  --chunk_size 16 \
  --sink_token 16 \
  --local_token 128 \
  --top_retrieval_heads 20 \
  --per_head_top_group 15
```

**Evaluation:**

```bash
python eval.py \
  --model gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --attn_type evosparse
```

`--dataset_path` defaults to `THUDM/LongBench` and may instead point to a local LongBench dataset script. Supported attention modes are `dense`, `evosparse`, the two component ablations, and the 32/64-block EvoSparse variants.

### RULER

RULER retains its upstream data-generation and evaluation structure. The following example prepares an **8K `niah_single_1` dataset**, runs EvoSparse, and evaluates the predictions.

**Data preparation:**

```bash
cd experiments/ruler

python data/prepare.py \
  --save_dir ./data/generated \
  --benchmark synthetic \
  --task niah_single_1 \
  --subset validation \
  --tokenizer_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --tokenizer_type hf \
  --model_template_type meta-chat \
  --max_seq_length 8192 \
  --num_samples 50
```

**Inference:**

```bash
torchrun --nproc_per_node=1 pred/run.py \
  --data_dir ./data/generated \
  --save_dir ./predictions/evosparse \
  --benchmark synthetic \
  --task niah_single_1 \
  --subset validation \
  --server_type hf \
  --model_name_or_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --max_seq_length 8192 \
  --attn_type evosparse \
  --budget 4096 \
  --top_head 20 \
  --top_group 15
```

**Evaluation:**

```bash
python eval/evaluate.py \
  --data_dir ./predictions/evosparse \
  --benchmark synthetic
```

## Citation

If you find EvoSparse useful in your research, please cite our ACL 2026 paper:

```bibtex
@inproceedings{han-etal-2026-evolving,
  title     = {Evolving Sparsity: Leveraging Token Importance Dynamics for Efficient LLM Decoding with Sparse Attention},
  author    = {Han, Ruizi and Zhang, Miao and Qiao, Ziyue and Nie, Liqiang},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2026},
  pages     = {11554--11566},
  doi       = {10.18653/v1/2026.acl-long.530}
}
```

## License

This project is released under the [Apache License 2.0](./LICENSE).
