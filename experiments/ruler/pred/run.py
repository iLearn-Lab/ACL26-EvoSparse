# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# service ssh restart

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import traceback
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from evosparse import (
    enable_evosparse,
    enable_evosparse_without_cross_layer,
    enable_evosparse_without_cross_step,
    reset_evosparse_state,
)
SERVER_TYPES = (
    'trtllm',
    'vllm',
    'openai',
    'gemini',
    'hf',
    'mamba',
)

ATTENTION_TYPES = (
    "evosparse",
    "dense",
    "evosparse-without-cross-layer",
    "evosparse-without-cross-step",
    "evosparse-32-block",
    "evosparse-64-block",
)


def read_manifest(path):
    """Read RULER JSONL without requiring the full NeMo package."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


REPO_ROOT = Path(__file__).resolve().parents[3]


def retrieval_head_config(model_name):
    return REPO_ROOT / "configs" / "retrieval_heads" / f"{model_name.split('/')[-1]}.json"


def load_retrieval_heads(model_name, count):
    """Load the highest-scoring retrieval heads for a model."""
    with open(retrieval_head_config(model_name), encoding="utf-8") as handle:
        scores = json.load(handle)
    ranked = sorted(
        ((name, float(np.mean(values))) for name, values in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [[int(index) for index in name.split("-")] for name, _ in ranked[:count]]


def configure_attention(model, parsed_args):
    """Configure one of the six public attention modes and return retrieval heads."""
    if parsed_args.attn_type == "dense":
        return []

    block_size = {
        "evosparse-32-block": 32,
        "evosparse-64-block": 64,
    }.get(parsed_args.attn_type, 16)
    local_tokens = 64 if parsed_args.budget <= 512 else 128
    if parsed_args.budget <= block_size + local_tokens:
        raise ValueError(
            "budget must be larger than the sink and local-window allocation"
        )

    patch = {
        "evosparse-without-cross-layer": enable_evosparse_without_cross_layer,
        "evosparse-without-cross-step": enable_evosparse_without_cross_step,
    }.get(parsed_args.attn_type, enable_evosparse)
    patch(
        model,
        group_size=block_size,
        total_budget=parsed_args.budget,
        sink_token=block_size,
        local_token=local_tokens,
        per_head_top_group=parsed_args.top_group,
        retrieval_head_num=parsed_args.top_head,
        enable_first_two_layers=False,
        decay=parsed_args.decay,
    )
    return load_retrieval_heads(parsed_args.model_name_or_path, parsed_args.top_head)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo',
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--K", type=int, default=9)
parser.add_argument("--L", type=int, default=200)
parser.add_argument("--max_seq_length", type=int, default=16384, help='max sequence length including all input tokens and generated tokens.')

parser.add_argument(
    "--attn_type",
    type=str,
    choices=ATTENTION_TYPES,
    default="evosparse",
)

parser.add_argument("--budget", type=int, default=4096)
parser.add_argument("--decay", type=float, default=0.996)
parser.add_argument("--top_head", type=int, default=20)
parser.add_argument("--top_group", type=int, default=15)

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini':
    args.threads = 1

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
DEVICE = torch.device("cuda", local_rank)

def main():
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'

    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'

    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    else:
        data = read_manifest(task_file)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    block_list = configure_attention(model, args)
    sparse_attention = args.attn_type != "dense"
    tokens_to_generate = config["tokens_to_generate"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    def get_output(idx, index, input, outputs, others, truncation, length):

        inputs = tokenizer(input, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        seq_len = inputs["input_ids"].shape[1] #pre-fill length




        chunk_size = 128
        past_key_values = None

        if sparse_attention:
            reset_evosparse_state(model)

        for i in range(0, seq_len, chunk_size):
            chunk = inputs.input_ids[:, i:i + chunk_size]
            # chunk = chunk.to(model.device)
            with torch.no_grad():
                if sparse_attention:
                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        is_prefilling=True
                    )
                else:
                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = output.past_key_values

        with torch.no_grad():
            # past_key_values = output.past_key_values
            # for input_id in q_input.input_ids[0]:
            #     output = model(
            #         input_ids=input_id.unsqueeze(0).unsqueeze(0),
            #         past_key_values=past_key_values,
            #         use_cache=True,
            #         # block_list=block_list,
            #         is_prefilling=False
            #     )
            #     past_key_values = output.past_key_valuesww

            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(tokens_to_generate - 1):
                if sparse_attention:
                    gen_outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                        block_list=block_list,
                        is_prefilling=False
                    )
                else:
                    gen_outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = gen_outputs.past_key_values
                pred_token_idx = (
                    gen_outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                )
                generated_content += [pred_token_idx.item()]

                # if pred_token_idx.item() == tokenizer.eos_token_id:
                #     break
                if pred_token_idx.item() in [128008, 128001, 128009]:
                    break

        generated_text = tokenizer.decode(generated_content, skip_special_tokens=True)

        pred = {'text': generated_text}

        if len(pred['text']) > 0:
            outputs_parallel[idx] = {
                'index': index,
                'pred': pred['text'],
                'input': input,
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
            }


    outputs_parallel = [{} for _ in range(len(data))]
    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):

            get_output(idx=idx,
                    index=data_point['index'],
                    input=data_point['input'],
                    outputs=data_point['outputs'],
                    others=data_point.get('others', {}),
                    truncation=data_point.get('truncation', -1),
                    length=data_point.get('length', -1))


            if len(outputs_parallel[idx]) > 0 and local_rank == 0:
                fout.write(json.dumps(outputs_parallel[idx]) + '\n')

    dist.destroy_process_group()

    # decode_tokens = llm.attention_server.decode_tokens
    # workload = llm.attention_server.workload
    # avg_workload = torch.tensor(workload / decode_tokens).to(llm.device)
    # dist.all_reduce(avg_workload,op=dist.ReduceOp.SUM)
    # world_size = dist.get_world_size()
    # if local_rank == 0:
if __name__ == '__main__':
    main()
#4.43.1
