# EvoSparse

Official implementation of **Evolving Sparsity: Leveraging Token Importance Dynamics for Efficient LLM Decoding with Sparse Attention**, published at ACL 2026.

EvoSparse exploits the evolution of token importance during autoregressive decoding through two complementary components:

- **Cross-Step Accumulation** tracks decayed token importance across decoding steps.
- **Cross-Layer Propagation** discovers query-aware blocks at retrieval heads and propagates the selected blocks to subsequent layers.

[[Paper](https://aclanthology.org/2026.acl-long.530/)] [[DOI](https://doi.org/10.18653/v1/2026.acl-long.530)]

## Repository structure

```text
ACL26-EvoSparse/
├── src/evosparse/             # shared EvoSparse implementation
├── configs/retrieval_heads/   # model-specific retrieval-head scores
├── experiments/pg19/          # PG-19 perplexity evaluation
├── experiments/ruler/         # RULER generation, inference, and evaluation
├── experiments/longbench/     # LongBench inference and evaluation
├── figures/                   # plotting scripts and generated figures
└── results/                   # local experiment outputs
```

## Environment

- Python 3.11.13
- PyTorch 2.8.0 with CUDA 12.8
- Transformers 4.52.4
- Datasets 3.6.0

Create the environment and install the project with:

```bash
conda create -n evosparse python=3.11 -y
conda activate evosparse

# Install the CUDA 12.8 build used in our experiments.
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install EvoSparse and the pinned benchmark dependencies from pyproject.toml.
pip install -e ".[experiments]"
```


## Method API

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

## PG-19 perplexity

Run commands from the repository root. The following configuration corresponds to a token budget of 4096, block size 16, 20 retrieval heads, and 15 candidate groups per retrieval head.

Dense attention:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/pg19/evaluate.py \
  --model_name_or_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --attention dense \
  --num_eval_tokens 32000 \
  --output_dir results/pg19_dense
```

EvoSparse:

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/pg19/evaluate.py \
  --model_name_or_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --attention evosparse \
  --budget 4096 \
  --block_size 16 \
  --retrieval_heads 20 \
  --groups_per_head 15 \
  --num_eval_tokens 32000 \
  --output_dir results/pg19_evosparse_b4096_bs16_h20_g15
```

## LongBench

```bash
cd experiments/longbench

CUDA_VISIBLE_DEVICES=0 python pred.py \
  --model gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --task qasper \
  --attn_type evosparse \
  --budget 4096 \
  --chunk_size 16 \
  --sink_token 16 \
  --local_token 128 \
  --top_retrieval_heads 20 \
  --per_head_top_group 15

python eval.py \
  --model gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --attn_type evosparse
```

`--dataset_path` defaults to `THUDM/LongBench` and may instead point to a local LongBench dataset script. Supported attention modes are `dense`, `evosparse`, the two component ablations, and the 32/64-block EvoSparse variants.

## RULER

RULER retains its upstream data-generation and evaluation structure. The following example prepares an 8K `niah_single_1` dataset, runs EvoSparse, and evaluates the predictions:

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

python eval/evaluate.py \
  --data_dir ./predictions/evosparse \
  --benchmark synthetic
```


## Citation

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

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
