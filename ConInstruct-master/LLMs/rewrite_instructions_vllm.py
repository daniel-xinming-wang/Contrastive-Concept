# -*- coding: UTF-8 -*-

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from gpt4 import idx_2_conflict_type, list_of_ints, read_input


REWRITE_PROMPT_TEMPLATE = """You will rewrite one instruction by replacing exactly one constraint.

Original Instruction:
{original_instruction}

Original Constraint to replace:
{org_constraint}

New Constraint to use as replacement:
{new_constraint}

Rewrite the Original Instruction so that:
1. It preserves all requirements from the Original Instruction except the Original Constraint.
2. It replaces only the Original Constraint.
3. It does not preserve, restate, or imply the Original Constraint.
4. It does not include both the Original Constraint and the New Constraint.
5. It keeps the wording as close as possible to the Original Instruction.
6. It outputs only the rewritten instruction, with no explanation, preface, bullet list, or markdown fence.

Rewritten Instruction:"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use a local vLLM model to rewrite ConInstruct instructions so each "
            "rewritten instruction contains only the new side of a conflict."
        )
    )
    parser.add_argument('--model', type=str, default='google/gemma-4-31B')
    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--input_filename', type=str, default='./datasets/conflict_instruction.jsonl')
    parser.add_argument(
        '--output_filename',
        type=str,
        default=None,
        help='Defaults to outputs/coninstruct_rewrites/{model}/new_instruction_rewrites.jsonl.',
    )
    parser.add_argument(
        '--conflict_type_idx',
        default='1-2-3-4-5-6-7-8-9',
        help='Conflict type ids to rewrite, joined by "-".',
    )
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--dtype', type=str, default='auto')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.conflict_type_idx_list = list_of_ints(args.conflict_type_idx)
    return args


def model_path(args):
    return os.path.join(args.cache_dir, args.model) if args.cache_dir else args.model


def model_name_for_path(model_name):
    return model_name.strip('/').replace('/', '__')


def default_output_filename(args):
    return (
        Path('./outputs/coninstruct_rewrites')
        / model_name_for_path(args.model)
        / 'new_instruction_rewrites.jsonl'
    )


def load_existing_keys(output_filename):
    existing = set()
    path = Path(output_filename)
    if not path.exists():
        return existing
    with path.open('r', encoding='utf-8') as fr:
        for line in fr:
            if not line.strip():
                continue
            item = json.loads(line)
            existing.add((item['sample_id'], item['conflict_type_idx']))
    return existing


def clean_rewrite(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    prefixes = [
        'Rewritten Instruction:',
        'Rewritten instruction:',
        'Here is the rewritten instruction:',
        'Here is the revised instruction:',
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def format_prompt(tokenizer, prompt):
    messages = [
        {'role': 'system', 'content': 'You are a careful instruction editor.'},
        {'role': 'user', 'content': prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_items(args, tokenizer, existing_keys):
    items = []
    for sample_id, instruction_dict in enumerate(read_input(args.input_filename), start=1):
        if args.max_samples is not None and sample_id > args.max_samples:
            break

        original_instruction = instruction_dict['instruction']
        for conflict_type_idx in args.conflict_type_idx_list:
            conflict_type = idx_2_conflict_type[conflict_type_idx]
            if conflict_type not in instruction_dict:
                continue
            if not args.overwrite and (sample_id, conflict_type_idx) in existing_keys:
                continue

            conflict_info = instruction_dict[conflict_type]
            prompt = REWRITE_PROMPT_TEMPLATE.format(
                original_instruction=original_instruction,
                org_constraint=conflict_info['org_constraint'],
                new_constraint=conflict_info['new_constraint'],
            )
            items.append({
                'sample_id': sample_id,
                'conflict_type_idx': conflict_type_idx,
                'conflict_type': conflict_type,
                'original_instruction': original_instruction,
                'org_constraint': conflict_info['org_constraint'],
                'new_constraint': conflict_info['new_constraint'],
                'rewrite_prompt': prompt,
                'formatted_prompt': format_prompt(tokenizer, prompt),
            })
    return items


def generate_rewrites(llm, sampling_params, items, batch_size):
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        prompts = [item['formatted_prompt'] for item in batch_items]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for item, output in zip(batch_items, outputs):
            raw_text = output.outputs[0].text
            item['raw_rewrite'] = raw_text
            item['new_instruction'] = clean_rewrite(raw_text)
            del item['formatted_prompt']
            yield item


def main():
    args = parse_args()
    output_filename = Path(args.output_filename) if args.output_filename else default_output_filename(args)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = set() if args.overwrite else load_existing_keys(output_filename)
    if args.overwrite and output_filename.exists():
        output_filename.unlink()

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

    items = build_items(args, tokenizer, existing_keys)
    print(f'Rewriting {len(items)} instructions')
    with output_filename.open('a', encoding='utf-8') as fw:
        for item in generate_rewrites(llm, sampling_params, items, args.batch_size):
            item['rewrite_model'] = args.model
            fw.write(json.dumps(item, ensure_ascii=False) + '\n')
            fw.flush()

    print(f'Wrote rewrites to {output_filename}')


if __name__ == '__main__':
    main()
