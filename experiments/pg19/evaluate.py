import torch
from tqdm import tqdm
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
import argparse
import json
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from evosparse import enable_evosparse

def get_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Perplexity of Llama on PG19 Dataset"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        choices=["togethercomputer/Llama-2-7B-32K-Instruct","gradientai/Llama-3-8B-Instruct-Gradient-1048k"],
        help="The name or path of the model to evaluate. For PPL on raw text, using a Base model is recommended.",
    )
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pg19_results_llama3_ablation",
        help="Directory to save logs and results.",
    )
    parser.add_argument(
        "--num_eval_tokens",
        type=int,
        default=32000,
        help="The number of tokens to evaluate for a quick test. Set to -1 to evaluate all tokens in the limited docs.",
    )
    parser.add_argument(
        "--limit_docs",
        type=int,
        default=1,
        help="Limit the number of documents from the dataset to process. The full test set is 100 documents.",
    )
    parser.add_argument(
        "--use_fp32",
        action="store_true",
        help="Force use of float32 for maximum stability (at the cost of memory and speed).",
    )
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--sink_tokens", type=int, default=16)
    parser.add_argument("--local_tokens", type=int, default=128)
    parser.add_argument("--retrieval_heads", type=int, default=20)
    parser.add_argument("--groups_per_head", type=int, default=15)
    parser.add_argument("--decay", type=float, default=0.996)
    parser.add_argument("--retrieval_head_config", default=None)
    parser.add_argument(
        "--attention",
        choices=["dense", "evosparse"],
        default="evosparse",
        help="Attention implementation to evaluate.",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name_or_path, args):
    """
    Loads the model and tokenizer with robust dtype selection.
    """
    print(f"Loading model from {model_name_or_path} ...")

    if args.use_fp32:
        selected_dtype = torch.float32
        print("Forcing float32 as requested.")
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        selected_dtype = torch.bfloat16
        print("bfloat16 is supported, using it for optimal performance and stability.")
    else:
        selected_dtype = torch.float16
        print(
            "Warning: bfloat16 not supported. Falling back to float16. This may cause numerical instability (NaNs)."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, attn_implementation="eager",device_map="auto", torch_dtype=selected_dtype
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def main():
    """
    Main function to run the evaluation.
    """
    args = get_args()

    print("Loading pg19 test dataset...")
    try:
        data = load_dataset("emozilla/pg19-test", split="test")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args)

    if args.attention == "evosparse":
        enable_evosparse(
            model,
            group_size=args.block_size,
            total_budget=args.budget,
            sink_token=args.sink_tokens,
            local_token=args.local_tokens,
            per_head_top_group=args.groups_per_head,
            retrieval_head_num=args.retrieval_heads,
            enable_first_two_layers=False,
            decay=args.decay,
        )

    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []

    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "log.txt")
    print(f"Logging individual token NLLs to: {log_file_path}")
    log_f = open(log_file_path, "w")

    num_total_eval_tokens = 0
    docs_to_process = data.select(range(args.limit_docs))
    eval_start = time.perf_counter()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    head_config = args.retrieval_head_config or os.path.join(
        repo_root,
        "configs",
        "retrieval_heads",
        f"{args.model_name_or_path.split('/')[-1]}.json",
    )
    with open(head_config, "r") as file:
        stable_block_list =  json.loads(file.readline())
    stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
    stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
    block_list = [
        [int(index) for index in item[0].split("-")]
        for item in stable_block_list[: args.retrieval_heads]
    ]

    for doc_idx, doc in enumerate(docs_to_process):
        text = doc["text"]
        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        print(
            f"\nProcessing document {doc_idx + 1}/{len(docs_to_process)} with sequence length: {seq_len}"
        )

        past_key_values = None
        pbar = tqdm(
            range(seq_len - 1),
            desc="Evaluating PPL",
            disable=args.no_progress,
        )

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(model.device)
            with torch.no_grad():
                if args.attention == "evosparse":
                    outputs = model.model(
                        input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        block_list=block_list,
                        is_prefilling=False,
                    )
                    logits = model.lm_head(outputs.last_hidden_state).view(
                        -1, model.config.vocab_size
                    )
                else:
                    outputs = model(
                        input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = (
                    encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                )
                neg_log_likelihood = loss_fn(logits, label)

            if torch.isnan(neg_log_likelihood) or torch.isinf(neg_log_likelihood):
                print(
                    f"\nWarning: Encountered NaN/Inf at token {idx+1} for document {doc_idx+1}. Skipping rest of the document to prevent cache poisoning."
                )
                break

            # Accumulate in FP32 so the final mean is not quantized to BF16.
            nlls.append(neg_log_likelihood.float())
            pbar.set_description(
                f"Token {idx+1}/{seq_len-1} | NLL: {neg_log_likelihood.item():.2f}"
            )
            log_f.write(f"{neg_log_likelihood.item()}\n")
            log_f.flush()

            num_total_eval_tokens += 1
            if (
                args.num_eval_tokens != -1
                and num_total_eval_tokens >= args.num_eval_tokens
            ):
                break

        if args.num_eval_tokens != -1 and num_total_eval_tokens >= args.num_eval_tokens:
            print(
                f"\nReached token limit of {args.num_eval_tokens}. Stopping evaluation."
            )
            break

    log_f.close()

    if not nlls:
        print("No tokens were evaluated. Cannot calculate PPL.")
        return

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\n--- Final Result ---")
    print(
        f"Perplexity on pg19 (evaluated on {len(nlls)} valid tokens): {ppl.item():.4f}"
    )

    ppl_file_path = os.path.join(args.output_dir, "ppl.txt")
    with open(ppl_file_path, "w") as f:
        f.write(f"{ppl.item()}\n")
    print(f"Perplexity score saved to: {ppl_file_path}")
    elapsed = time.perf_counter() - eval_start
    summary = {
        "attention": args.attention,
        "perplexity": ppl.item(),
        "evaluated_tokens": len(nlls),
        "elapsed_seconds": elapsed,
        "tokens_per_second": len(nlls) / elapsed,
        "block_size": args.block_size if args.attention == "evosparse" else None,
        "budget": args.budget if args.attention == "evosparse" else None,
        "retrieval_heads": args.retrieval_heads if args.attention == "evosparse" else None,
        "groups_per_head": args.groups_per_head if args.attention == "evosparse" else None,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Evaluation throughput: {summary['tokens_per_second']:.2f} tokens/s")


if __name__ == "__main__":
    main()
