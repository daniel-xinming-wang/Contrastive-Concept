# -*- coding: UTF-8 -*-
# run with python version>3.6
# pip install openai

api_key_dict= {'key': 'your_api_key'}

import argparse
import os
import json
import random
import time
import requests

class OpenAIGPT4:
    def __init__(self, api_key: str, model: str, max_tokens: int = 256, 
                 temperature: float =  0, wait_time: int  = 10, retry_times: int  = 5):
        
        self.api_key = api_key
        self.model = model

        self.max_tokens = max_tokens
        self.temperature  = temperature
        self.wait_time = wait_time
        self.retry_times = retry_times
        
        self.headers = {"Content-Type": "application/json", 
                        "Authorization": f"Bearer {api_key}"}
    

    def generate(self, user_msg: str)->str:
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_msg}
            ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": 1
        }
        retry = True
        retry_times = 0
        while retry and retry_times < self.retry_times:
            # response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            try:
                response = requests.post("https://api.chatanywhere.tech/v1/chat/completions", headers=self.headers, json=payload)
                print(response)
                response.encoding = 'utf-8'
                if response.status_code == 200:
                    response_data = response.json()
                    # print(response_data["usage"])
                    text =  response_data["choices"][0]["message"]["content"]
                    # text = text.encode('utf-8').decode('unicode_escape')
                    return text
            except:
                    print(f"Fail to connect to OpenAI API. Retrying ...")
                    time.sleep(self.wait_time)
                    retry_times += 1
        return "Fail to connect to OpenAI API."


def read_input(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            d = json.loads(line)
            yield d
def inject_conflict_constraints(instruction_dict, conflict_type_idx_list, idx_2_conflict_type, conflict_type_list=None,
                                shuffle_new_constraint=False, new_constraint_position='after'):
    """_summary_
    This function is used to add conflicting constraints into the given instruction.
    Args:
        instruction_dict (_type_): _description_
        conflict_type_idx_list (_type_): list of int
        idx_2_conflict_type (_type_): a dict, which maps idxs to conflict types.
        shuffle_new_constraint (bool, optional): Whether to shuffle the new constraints or not. Defaults to False.
        new_constraint_position (str, optional): The position of the new constraints relative to the original instruction. Defaults to 'after'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: The revised instruction, which contains multiple conflicting constraints.
    """
    instruction = instruction_dict["instruction"]
    new_constraint_list = []
    conflict_dict = {}
    flag = 0
    if conflict_type_list is None:
        for idx in conflict_type_idx_list:
            conflict_type = idx_2_conflict_type[idx]
            if conflict_type not in instruction_dict:
                flag = 1
                break
            new_constraint = instruction_dict[conflict_type]['new_constraint']
            new_constraint_list.append(new_constraint.strip())
            conflict_dict[conflict_type] = instruction_dict[conflict_type]
    else:
        for conflict_type in conflict_type_list:
            new_constraint = instruction_dict[conflict_type]['new_constraint']
            new_constraint_list.append(new_constraint.strip())
            conflict_dict[conflict_type] = instruction_dict[conflict_type]
            
    if flag:
        return None, None
    if shuffle_new_constraint:
        random.shuffle(new_constraint_list)
    if new_constraint_position == 'after':
        instruction = instruction.strip() +" " + " ".join(new_constraint_list)
    elif new_constraint_position == 'before':
        instruction =  " ".join(new_constraint_list) +" " + instruction.strip()
    else:
        raise ValueError()
    return instruction, conflict_dict
        
conflict_type_2_idx = {
    "Conflicts between Content Constraints": 1,
    "Conflicts between Keyword Constraints": 2,
    "Conflicts between Keyword Constraints and Phrase Constraints": 3,
    "Conflicts between Phrase Constraints": 4,
    "Conflicts between Length Constraints": 5,
    "Conflicts between Format Constraints": 6,
    "Conflicts between Style Constraints": 7,
    "Conflicts between Phrase Constraints and Content Constraints": 8,
    "Conflicts between Phrase Constraints and Style Constraints": 9
}

idx_2_conflict_type = {
    1: "Conflicts between Content Constraints",
    2: "Conflicts between Keyword Constraints",
    3: "Conflicts between Keyword Constraints and Phrase Constraints",
    4: "Conflicts between Phrase Constraints",
    5: "Conflicts between Length Constraints",
    6: "Conflicts between Format Constraints",
    7: "Conflicts between Style Constraints",
    8: "Conflicts between Phrase Constraints and Content Constraints",
    9: "Conflicts between Phrase Constraints and Style Constraints"
}

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split('-')))

def str_2_bool(arg):
    if arg.lower() in ['1', 'yes', 'true', 'y', 't']:
        return True
    if arg.lower() in ['0', 'no', 'false', 'n', 'f']:
        return False

def get_argument(parser):
    parser.add_argument('--max_tokens', type=int, default=2048, 
                        help='The number of max tokens for the generated output.')   
    parser.add_argument('--temperature', type=float, default=0)   
    parser.add_argument('--wait_time', type=int, default=10, 
                        help='Retry your request after a specified seconds.')  
    parser.add_argument('--retry_times', type=int, default=10, 
                        help='The maximum number of retry times.') 
    parser.add_argument('--conflict_type_idx', default="1-2", 
                        help='The type of conflicting constraints added to the original instruction.')
    parser.add_argument('--shuffle_new_constraint', default="true", type=str_2_bool, 
                        help='Whether to shuffle the new constraints or not. \
                        This parameter will have no effect if the length of conflict_type_idx_list is 1.')
    parser.add_argument('--new_constraint_position', default="after", 
                        help='The position of the new constraints relative to the original instruction.')
    parser.add_argument('--task', type=str, default='conflict_resolution', 
                        choices=['instruction_following', 'conflict_resolution_density_expected_behavior', 'conflict_resolution', 'conflict_detection',
                                 'conflict_resolution_density', 'conflict_detection_density', "conflict_detection_prob"])
    parser.add_argument('--num_conflict', type=int, default=0, help='Only effective when task=conflict_detection_density')
    parser.add_argument('--input_filename', type=str, default='./datasets/conflict_instruction.jsonl') 
    parser.add_argument('--restart_idx', type=int, default=0)
    args = parser.parse_args()
    args.conflict_type_idx_list = list_of_ints(args.conflict_type_idx)
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openai.')
    parser.add_argument('--api_key', type=str, required=False, default='key') 
    parser.add_argument('--model', type=str, default="gpt-4-turbo-2024-04-09", 
                        choices=['o1-preview-2024-09-12',
                                 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 
                                 'gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18', 
                            ])  
    args = get_argument(parser)
    print(args)
    random.seed(42)
    if args.task in ["conflict_resolution",  "conflict_detection"]:
        if args.temperature>0:
            output_dir = f'./outputs/{args.task}/{args.model}_{args.temperature}/{args.conflict_type_idx}'
        else:
            output_dir = f'./outputs/{args.task}/{args.model}/{args.conflict_type_idx}'
        if args.new_constraint_position == 'before':
            output_dir = f'./outputs/{args.task}/{args.model}/new_org_{args.conflict_type_idx}'
    elif args.task =="instruction_following":
        output_dir = f'./outputs/{args.task}/{args.model}'
    elif args.task in ["conflict_resolution_density", "conflict_detection_density", "conflict_resolution_density_expected_behavior", "conflict_detection_prob"]:
        output_dir = f'./outputs/{args.task}/{args.model}/{args.num_conflict}_conflict'
        if args.new_constraint_position == 'before':
            output_dir = f'./outputs/{args.task}/{args.model}/new_org_{args.num_conflict}_conflict'
        if args.num_conflict>0:
            conflict_lists = json.load(open(f'./datasets/conflict_density/{args.num_conflict}_conflict.json','r'))
    else:
        pass
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    llm = OpenAIGPT4(api_key=api_key_dict[args.api_key], model=args.model,
                      temperature=args.temperature, max_tokens=args.max_tokens,
                      wait_time=args.wait_time, retry_times=args.retry_times)
    i = 0
    for instruction_dict in read_input(args.input_filename):
        i+=1
        if i<=args.restart_idx:
            continue
        # if i>1:
            # break
        print(i)
        if args.task == 'instruction_following' or args.conflict_type_idx=="0" or args.num_conflict==0:
            instruction = instruction_dict["instruction"]
        elif args.task in ["conflict_resolution", "conflict_detection"]:
            instruction, conflict_dict = inject_conflict_constraints(instruction_dict, args.conflict_type_idx_list, 
                                                                    idx_2_conflict_type, conflict_type_list=None,
                                                                    shuffle_new_constraint=args.shuffle_new_constraint, 
                                                                    new_constraint_position=args.new_constraint_position)
        elif args.task in ["conflict_resolution_density", "conflict_detection_density", "conflict_resolution_density_expected_behavior", "conflict_detection_prob"]:
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
        if args.task in ["conflict_resolution_density_expected_behavior"]:
            with open('./prompts/instruction_following_expected_behavior.txt', 'r', encoding='utf-8') as fr:
                constraint_prompt = fr.read()
                instruction = constraint_prompt.replace('##instruction##', instruction)

        if args.task in ["conflict_detection_prob"]:
            with open('./prompts/conflict_detection_prob.txt', 'r', encoding='utf-8') as fr:
                constraint_prompt = fr.read()
                instruction = constraint_prompt.replace('##instruction##', instruction)
        
        response = llm.generate(user_msg=instruction)
        if args.task == 'instruction_following' or args.conflict_type_idx=="0" or args.num_conflict==0:
            output_dict = { "model": args.model,
                            "instruction": instruction,
                            "response": response}
        elif args.task in ["conflict_resolution", "conflict_resolution_density", "conflict_detection", "conflict_detection_density", "conflict_resolution_density_expected_behavior", "conflict_detection_prob"]:
            output_dict = { "model": args.model,
                            "instruction": instruction,
                            "response": response,
                            "conflicts": conflict_dict}
        else:
            pass
        with open(f"{output_dir}/{i}.json", 'w', encoding='utf-8') as fw:
            json.dump(output_dict, fw, indent = 4)
