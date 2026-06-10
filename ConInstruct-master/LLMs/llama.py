# This script supports open-sourced llms
import argparse
import os
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from gpt4 import read_input, inject_conflict_constraints, get_argument, idx_2_conflict_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open-sourced models.')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-32B-Instruct", 
                        choices=['meta-llama/Llama-3.3-70B-Instruct','meta-llama/Meta-Llama-3.1-8B-Instruct', 
                                 "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
                                 'mistralai/Ministral-8B-Instruct-2410', 'Qwen/Qwen2.5-0.5B-Instruct',
                                 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct',
                                 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct',
                                 'Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-72B-Instruct'])  
    parser.add_argument('--cache_dir', type=str, default="/home/hexw/code/HuggingfaceModels")
    parser.add_argument('--quantization4', '-q4', action='store_true')
    parser.add_argument('--quantization8', '-q8', action='store_true')
    args = get_argument(parser)
    print(args)
    random.seed(42)
    if args.task in ["conflict_resolution",  "conflict_detection"]:
        output_dir = f'./outputs/{args.task}/{args.model}/{args.conflict_type_idx}'
        if args.new_constraint_position == 'before':
            output_dir = f'./outputs/{args.task}/{args.model}/new_org_{args.conflict_type_idx}'
    elif args.task == 'instruction_following':
        output_dir = f'./outputs/{args.task}/{args.model}'
    elif args.task in ["conflict_resolution_density", "conflict_detection_density"]:
        output_dir = f'./outputs/{args.task}/{args.model}/{args.num_conflict}_conflict'
        if args.new_constraint_position == 'before':
            output_dir = f'./outputs/{args.task}/{args.model}/new_org_{args.num_conflict}_conflict'
        conflict_lists = json.load(open(f'./datasets/conflict_density/{args.num_conflict}_conflict.json','r'))
    else:
        pass
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.quantization4 or args.quantization8:
        if args.quantization8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config = None

    model_path = os.path.join(args.cache_dir, args.model) if args.cache_dir is not None else args.model
    model = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        device_map="auto", 
                        torch_dtype=torch.bfloat16, 
                        quantization_config=quantization_config
                        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    i = 0      
    for instruction_dict in read_input(args.input_filename):
        i+=1
        if i<=args.restart_idx:
            continue
        print(i)
        if args.task == 'instruction_following' or args.conflict_type_idx=="0":
            instruction = instruction_dict["instruction"]
        elif args.task in ["conflict_resolution",  "conflict_detection"]:
            instruction, conflict_dict = inject_conflict_constraints(instruction_dict, args.conflict_type_idx_list, 
                                                                    idx_2_conflict_type, 
                                                                    shuffle_new_constraint=args.shuffle_new_constraint, 
                                                                    new_constraint_position=args.new_constraint_position)
        elif args.task in ["conflict_resolution_density", "conflict_detection_density"]:
            conflict_type_list = conflict_lists[i-1]
            instruction, conflict_dict = inject_conflict_constraints(instruction_dict, None, None,
                                                                    conflict_type_list=conflict_type_list, 
                                                                    shuffle_new_constraint=args.shuffle_new_constraint, 
                                                                    new_constraint_position=args.new_constraint_position)
        else:
            pass
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

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, 
                                        max_new_tokens=args.max_tokens, 
                                        do_sample=False
                                        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_filename = f"{output_dir}/{i}.json"
        if args.task == 'instruction_following' or args.conflict_type_idx=="0":
            output_dict = { "model": args.model,
                            "instruction": instruction,
                            "response": response}
        elif args.task in ["conflict_resolution", "conflict_resolution_density", "conflict_detection", "conflict_detection_density"]:
            output_dict = { "model": args.model,
                            "instruction": instruction,
                            "response": response,
                            "conflicts": conflict_dict}
        else:
            pass
        with open(output_filename, 'w', encoding='utf-8') as fw:
            json.dump(output_dict, fw, indent = 4)
            
