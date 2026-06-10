# This script supports open-sourced llms with vLLM
import argparse
import os
import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from gpt4 import read_input, inject_conflict_constraints, get_argument, idx_2_conflict_type


def build_output_dir(args):
    if args.task in ["conflict_resolution", "conflict_detection"]:
        output_dir = f'./outputs/{args.task}/{args.model}/{args.conflict_type_idx}'
        if args.new_constraint_position == 'before':
            output_dir = f'./outputs/{args.task}/{args.model}/new_org_{args.conflict_type_idx}'
    elif args.task == 'instruction_following':
        output_dir = f'./outputs/{args.task}/{args.model}'
    elif args.task in ["conflict_resolution_density", "conflict_detection_density"]:
        output_dir = f'./outputs/{args.task}/{args.model}/{args.num_conflict}_conflict'
        if args.new_constraint_position == 'before':
            output_dir = f'./outputs/{args.task}/{args.model}/new_org_{args.num_conflict}_conflict'
    else:
        raise ValueError(f"Unsupported task for llama_vllm.py: {args.task}")
    return output_dir


def build_generation_items(args, tokenizer):
    if args.task in ["conflict_resolution_density", "conflict_detection_density"]:
        conflict_lists = json.load(open(f'./datasets/conflict_density/{args.num_conflict}_conflict.json', 'r'))
    else:
        conflict_lists = None

    items = []
    i = 0
    for instruction_dict in read_input(args.input_filename):
        i += 1
        if i <= args.restart_idx:
            continue
        print(i)
        conflict_dict = None
        if args.task == 'instruction_following' or args.conflict_type_idx == "0":
            instruction = instruction_dict["instruction"]
        elif args.task in ["conflict_resolution", "conflict_detection"]:
            instruction, conflict_dict = inject_conflict_constraints(
                instruction_dict,
                args.conflict_type_idx_list,
                idx_2_conflict_type,
                shuffle_new_constraint=args.shuffle_new_constraint,
                new_constraint_position=args.new_constraint_position
            )
        elif args.task in ["conflict_resolution_density", "conflict_detection_density"]:
            conflict_type_list = conflict_lists[i - 1]
            instruction, conflict_dict = inject_conflict_constraints(
                instruction_dict,
                None,
                None,
                conflict_type_list=conflict_type_list,
                shuffle_new_constraint=args.shuffle_new_constraint,
                new_constraint_position=args.new_constraint_position
            )
        else:
            raise ValueError(f"Unsupported task for llama_vllm.py: {args.task}")

        if instruction is None:
            continue
        if args.task in ["conflict_detection", "conflict_detection_density"]:
            with open('./prompts/conflict_detection.txt', 'r', encoding='utf-8') as fr:
                constraint_prompt = fr.read()
                instruction = constraint_prompt.replace('##instruction##', instruction)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        items.append({
            "idx": i,
            "instruction": instruction,
            "prompt": prompt,
            "conflicts": conflict_dict,
        })
    return items


def write_outputs(args, output_dir, items, responses):
    for item, response in zip(items, responses):
        output_filename = f"{output_dir}/{item['idx']}.json"
        if args.task == 'instruction_following' or args.conflict_type_idx == "0":
            output_dict = {
                "model": args.model,
                "instruction": item["instruction"],
                "response": response,
            }
        elif args.task in ["conflict_resolution", "conflict_resolution_density", "conflict_detection", "conflict_detection_density"]:
            output_dict = {
                "model": args.model,
                "instruction": item["instruction"],
                "response": response,
                "conflicts": item["conflicts"],
            }
        else:
            raise ValueError(f"Unsupported task for llama_vllm.py: {args.task}")
        with open(output_filename, 'w', encoding='utf-8') as fw:
            json.dump(output_dict, fw, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open-sourced models with vLLM.')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument('--cache_dir', type=str, default="/home/hexw/code/HuggingfaceModels")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--dtype', type=str, default="bfloat16")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument(
        '--enable-thinking',
        action='store_true',
        dest='enable_thinking',
        help='Enable Qwen3.5 thinking mode in the chat template. Default is non-thinking mode.',
    )
    args = get_argument(parser)
    print(args)
    random.seed(42)

    output_dir = build_output_dir(args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(args.cache_dir, args.model) if args.cache_dir else args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    items = build_generation_items(args, tokenizer)
    for start in range(0, len(items), args.batch_size):
        batch_items = items[start:start + args.batch_size]
        prompts = [item["prompt"] for item in batch_items]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        responses = [output.outputs[0].text for output in outputs]
        write_outputs(args, output_dir, batch_items, responses)
