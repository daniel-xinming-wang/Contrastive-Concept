# -*- coding: UTF-8 -*-

import argparse
import json
import os
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluation import (
    extract_label,
    extract_label_conflict_resolution_behavior,
    read_input,
)


def str_2_bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ['1', 'yes', 'true', 'y', 't']:
        return True
    if arg.lower() in ['0', 'no', 'false', 'n', 'f']:
        return False
    raise ValueError(f"Cannot parse boolean value: {arg}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ConInstruct outputs with a local vLLM judge.')
    parser.add_argument('--judge_model', type=str, default='google/gemma-4-31B')
    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--dtype', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument(
        '--task',
        type=str,
        default='conflict_resolution',
        choices=['conflict_resolution', 'conflict_resolution_behavior'],
    )
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--restart_idx', type=int, default=0)
    parser.add_argument(
        '--include_judge_in_output_dir',
        type=str_2_bool,
        default=False,
        help='Whether to include the judge model name in evaluation_outputs.',
    )
    return parser.parse_args()


def model_path(args):
    return os.path.join(args.cache_dir, args.judge_model) if args.cache_dir else args.judge_model


def judge_name_for_path(judge_model):
    return judge_model.strip('/').replace('/', '__')


def build_output_dir(args):
    output_dir = args.input_dir.replace('outputs', 'evaluation_outputs')
    if args.task == 'conflict_resolution_behavior':
        output_dir = output_dir.replace('conflict_resolution_density', 'conflict_resolution_density/behavior')
        if 'conflict_resolution_density' not in output_dir:
            output_dir = output_dir.replace('conflict_resolution', 'conflict_resolution/behavior')

    if args.include_judge_in_output_dir:
        output_dir = os.path.join(output_dir, f"judge_{judge_name_for_path(args.judge_model)}")
    return output_dir


def format_judge_prompt(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_responses(llm, sampling_params, prompts, batch_size):
    responses = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        responses.extend(output.outputs[0].text for output in outputs)
    return responses


def evaluate_conflict_resolution(args, llm, tokenizer, sampling_params, output_dir):
    assert 'conflict_resolution' in args.input_dir
    with open('./prompts/conflict_resolution_pair_evaluation.txt', 'r', encoding='utf-8') as fr:
        prompt_template = fr.read()

    eval_items = []
    for i, data_dict in read_input(args.input_dir):
        if i <= args.restart_idx or data_dict is None:
            continue
        response_dict = {}
        prompt_tmp = prompt_template.replace('##model_response##', data_dict['response'])
        conflict_dict = data_dict['conflicts']
        for conflict_type in conflict_dict:
            response_dict[conflict_type] = {}
            new_constraint = conflict_dict[conflict_type]['new_constraint']
            org_constraint = conflict_dict[conflict_type]['org_constraint']
            if random.random() > 0.5:
                prompt = prompt_tmp.replace('##instruction1##', org_constraint).replace('##instruction2##', new_constraint)
                response_dict[conflict_type]['instruction_order'] = 'org_new'
            else:
                prompt = prompt_tmp.replace('##instruction1##', new_constraint).replace('##instruction2##', org_constraint)
                response_dict[conflict_type]['instruction_order'] = 'new_org'
            response_dict[conflict_type]['new_constraint'] = new_constraint
            response_dict[conflict_type]['org_constraint'] = org_constraint
            eval_items.append({
                'idx': i,
                'conflict_type': conflict_type,
                'prompt': format_judge_prompt(tokenizer, prompt),
            })
        response_dict['llm_response'] = data_dict['response']
        output_filename = os.path.join(output_dir, f'{i}.json')
        with open(output_filename, 'w', encoding='utf-8') as fw:
            json.dump(response_dict, fw, indent=4)

    responses = generate_responses(
        llm=llm,
        sampling_params=sampling_params,
        prompts=[item['prompt'] for item in eval_items],
        batch_size=args.batch_size,
    )

    for item, response in zip(eval_items, responses):
        output_filename = os.path.join(output_dir, f"{item['idx']}.json")
        with open(output_filename, 'r', encoding='utf-8') as fr:
            response_dict = json.load(fr)
        response_dict[item['conflict_type']]['evaluation_result'] = response
        with open(output_filename, 'w', encoding='utf-8') as fw:
            json.dump(response_dict, fw, indent=4)

    success_rate_dict = {}
    avg_new_constraint_sr = 0
    avg_org_constraint_sr = 0
    total = 0
    for i in range(1, 101):
        output_filename = f"{output_dir}/{i}.json"
        if not os.path.exists(output_filename):
            continue
        with open(output_filename, 'r', encoding='utf-8') as fr:
            response_dict = json.load(fr)
            for conflict_type in response_dict:
                if len(response_dict[conflict_type]) == 4:
                    if conflict_type not in success_rate_dict:
                        success_rate_dict[conflict_type] = {
                            'new_constraint_sr': 0,
                            'org_constraint_sr': 0,
                            'num': 0,
                        }
                    print(i)
                    label = extract_label(
                        response_dict[conflict_type]['evaluation_result'],
                        response_dict[conflict_type]['instruction_order'],
                    )
                    if label == 1:
                        success_rate_dict[conflict_type]['new_constraint_sr'] += 1
                        avg_new_constraint_sr += 1
                    elif label == 2:
                        success_rate_dict[conflict_type]['org_constraint_sr'] += 1
                        avg_org_constraint_sr += 1
                    total += 1
                    success_rate_dict[conflict_type]['num'] += 1

    for conflict_type in success_rate_dict:
        success_rate_dict[conflict_type]['new_constraint_sr'] /= success_rate_dict[conflict_type]['num']
        success_rate_dict[conflict_type]['org_constraint_sr'] /= success_rate_dict[conflict_type]['num']
    if total > 0:
        success_rate_dict['avg_new_constraint_sr'] = avg_new_constraint_sr / total
        success_rate_dict['avg_org_constraint_sr'] = avg_org_constraint_sr / total
    else:
        success_rate_dict['avg_new_constraint_sr'] = 0
        success_rate_dict['avg_org_constraint_sr'] = 0
    success_rate_dict['total'] = total
    success_rate_dict['judge_model'] = args.judge_model
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
        json.dump(success_rate_dict, fw, indent=4)


def evaluate_conflict_resolution_behavior(args, llm, tokenizer, sampling_params, output_dir):
    assert 'conflict_resolution' in args.input_dir
    if '0_conflict' in args.input_dir:
        with open('./prompts/instruction_following_behavior_evaluation.txt', 'r', encoding='utf-8') as fr:
            prompt_template = fr.read()
    else:
        with open('./prompts/conflict_resolution_behavior_evaluation.txt', 'r', encoding='utf-8') as fr:
            prompt_template = fr.read()

    eval_items = []
    for i, data_dict in read_input(args.input_dir):
        if i <= args.restart_idx or data_dict is None:
            continue
        prompt_tmp = prompt_template.replace('##model_response##', data_dict['response'])
        prompt = prompt_tmp.replace('##instruction##', data_dict['instruction'])
        eval_items.append({
            'idx': i,
            'prompt': format_judge_prompt(tokenizer, prompt),
            'instruction': data_dict['instruction'],
            'llm_response': data_dict['response'],
            'conflicts': data_dict.get('conflicts'),
        })

    responses = generate_responses(
        llm=llm,
        sampling_params=sampling_params,
        prompts=[item['prompt'] for item in eval_items],
        batch_size=args.batch_size,
    )

    for item, response in zip(eval_items, responses):
        response_dict = {
            'instruction': item['instruction'],
            'llm_response': item['llm_response'],
            'evaluation_result': response,
        }
        if item['conflicts'] is not None:
            response_dict['conflicts'] = item['conflicts']
        output_filename = os.path.join(output_dir, f"{item['idx']}.json")
        with open(output_filename, 'w', encoding='utf-8') as fw:
            json.dump(response_dict, fw, indent=4)

    result_dict = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        -1: 0,
        '1_ids': [],
        '2_ids': [],
        '3_ids': [],
        '4_ids': [],
        '5_ids': [],
        'judge_model': args.judge_model,
    }
    for i in range(1, 101):
        output_filename = f"{output_dir}/{i}.json"
        if not os.path.exists(output_filename):
            continue
        with open(output_filename, 'r', encoding='utf-8') as fr:
            response_dict = json.load(fr)
            print(i)
            label = extract_label_conflict_resolution_behavior(response_dict['evaluation_result'])
            result_dict[label] += 1
            if label == -1:
                result_dict['5_ids'].append(i)
            else:
                result_dict[f'{label}_ids'].append(i)
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
        json.dump(result_dict, fw, indent=4)


def main():
    args = parse_args()
    print(args)
    random.seed(42)

    output_dir = build_output_dir(args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)

    path = model_path(args)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.task == 'conflict_resolution':
        evaluate_conflict_resolution(args, llm, tokenizer, sampling_params, output_dir)
    elif args.task == 'conflict_resolution_behavior':
        evaluate_conflict_resolution_behavior(args, llm, tokenizer, sampling_params, output_dir)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == '__main__':
    main()
