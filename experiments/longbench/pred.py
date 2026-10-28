import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from evosparse import (
    enable_evosparse,
    enable_evosparse_without_cross_layer,
    enable_evosparse_without_cross_step,
    reset_evosparse_state,
)

ATTENTION_TYPES = [
    "evosparse",
    "dense",
    "evosparse-without-cross-layer",
    "evosparse-without-cross-step",
    "evosparse-32-block",
    "evosparse-64-block",
]


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn_type",
        type=str,
        choices=ATTENTION_TYPES,
        default="dense",
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--sink_token", type=int, default=16)
    parser.add_argument("--local_token", type=int, default=128)
    parser.add_argument("--top_retrieval_heads", type=int, default=10)
    parser.add_argument("--per_head_top_group", type=int, default=10)
    parser.add_argument("--enable_first_two_layers", action="store_true")
    parser.add_argument(
        "--dataset_path",
        default="THUDM/LongBench",
        help="Hugging Face dataset name or path to a local LongBench dataset script.",
    )

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "Llam3" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    block_list
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1

        # max simulation length is 100
        q_pos = max(len(prompt) - 100, q_pos)

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

        if "chatglm3" in model_name:
            input = prompt.to("cuda")
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to(
                "cuda"
            )
            q_input.input_ids = q_input.input_ids[:, 1:]

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            assert False
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            try:
                if "evosparse" in args.attn_type:
                    reset_evosparse_state(model)


                chunk_size = 128

                past_key_values = None
                for i in range(0, input.input_ids.shape[1], chunk_size):
                    chunk = input.input_ids[:, i:i + chunk_size]


                    with torch.no_grad():
                        output = model(
                            input_ids=chunk,
                            past_key_values=past_key_values,
                            use_cache=True,
                            is_prefilling=True
                        )
                        past_key_values = output.past_key_values

                        del output

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                with torch.no_grad():
                    for input_id in q_input.input_ids[0]:
                        output = model(
                            input_ids=input_id.unsqueeze(0).unsqueeze(0),
                            past_key_values=past_key_values,
                            use_cache=True,
                            block_list=block_list,
                            is_prefilling=False
                        )
                        past_key_values = output.past_key_values

                    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content = [pred_token_idx.item()]
                    for _ in range(max_gen - 1):
                        outputs = model(
                            input_ids=pred_token_idx,
                            past_key_values=past_key_values,
                            use_cache=True,
                            block_list=block_list,
                            is_prefilling=False
                        )

                        past_key_values = outputs.past_key_values
                        pred_token_idx = (
                            outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        )
                        generated_content += [pred_token_idx.item()]
                        if pred_token_idx.item() == tokenizer.eos_token_id:
                            break
            except torch.cuda.OutOfMemoryError as e:
                print(f"Caught CUDA OOM error: {e}")
                print("===================================================")
                print("Printing memory summary for GPU 0:")
                print("===================================================")
                print(torch.cuda.memory_summary(device=0))
                print("===================================================")
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            # )[0]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name_or_path, args):
    """
    Loads the model and tokenizer with robust dtype selection.
    """
    print(f"Loading model from {model_name_or_path} ...")

    #         "Warning: bfloat16 not supported. Falling back to float16. This may cause numerical instability (NaNs)."
    #     )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, attn_implementation="eager", device_map="auto", torch_dtype=torch.float16
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


if __name__ == "__main__":
    seed_everything(3407) #42
    args = parse_args()
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model

    model, tokenizer = load_model_and_tokenizer(model_name, args)
    if args.attn_type in {"evosparse", "evosparse-32-block", "evosparse-64-block"}:
        group_size = {
            "evosparse-32-block": 32,
            "evosparse-64-block": 64,
        }.get(args.attn_type, args.chunk_size)
        args.chunk_size = group_size
        enable_evosparse(
            model,
            group_size=group_size,
            total_budget=args.budget,
            sink_token=args.sink_token,
            local_token=args.local_token,
            per_head_top_group=args.per_head_top_group,
            enable_first_two_layers=args.enable_first_two_layers
        )
    elif args.attn_type == "evosparse-without-cross-layer":
        enable_evosparse_without_cross_layer(
            model,
            group_size=args.chunk_size,
            total_budget=args.budget,
            sink_token=args.sink_token,
            local_token=args.local_token,
            per_head_top_group=args.per_head_top_group,
            enable_first_two_layers=args.enable_first_two_layers
        )
    elif args.attn_type == "evosparse-without-cross-step":
        enable_evosparse_without_cross_step(
            model,
            group_size=args.chunk_size,
            total_budget=args.budget,
            sink_token=args.sink_token,
            local_token=args.local_token,
            per_head_top_group=args.per_head_top_group,
            enable_first_two_layers=args.enable_first_two_layers
        )
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset(args.dataset_path, f"{dataset}_e", split="test", trust_remote_code=True)
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
            if args.attn_type:
                out_path = (
                    f"pred_e/{model_name}/{dataset}-{args.attn_type}-budget{args.budget}.jsonl"
                )
            else:
                out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset(args.dataset_path, dataset, split="test", trust_remote_code=True)
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if not os.path.exists(f"pred/{model_name}/ours"):
                os.makedirs(f"pred/{model_name}/ours")
            if "evosparse" in args.attn_type:
                out_path = f"pred/{model_name}/ours/{dataset}-{args.attn_type}-budget{args.budget}-chunk_size{args.chunk_size}-sink{args.sink_token}-local{args.local_token}-top_retrieval_heads{args.top_retrieval_heads}-per_head_top_group{args.per_head_top_group}-enable_first_two_layers{args.enable_first_two_layers}.jsonl"  # ours
            else:
                out_path = f"pred/{model_name}/{dataset}-{args.attn_type}-{args.budget}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        print(f"task: {dataset}")
        if dataset == "narrativeqa":
            max_length = 28000
        print(f"max_length: {max_length}")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        with open(os.path.join(repo_root, "configs", "retrieval_heads", f"{model_name.split('/')[-1]}.json"), "r") as file:
            stable_block_list =  json.loads(file.readline())
        stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
        stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
        block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:args.top_retrieval_heads]

        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            block_list,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
