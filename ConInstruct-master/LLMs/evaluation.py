# -*- coding: UTF-8 -*-

api_key_dict= {'key': 'your_api_key'}
import argparse
import os
import json
import random
import sys
import time
import requests
from gpt4 import OpenAIGPT4, conflict_type_2_idx, idx_2_conflict_type, list_of_ints

def read_input(input_dir):
    for i in range(1, 101):
        filename = os.path.join(input_dir, f"{i}.json")
        if not os.path.exists(filename):
            yield i, None
            continue
        with open(filename, 'r', encoding='utf-8') as fr:
            d = json.load(fr)
            yield i, d
def extract_conflict_detection_prob(evaluation_result):
    if "[Probability]" in evaluation_result:
        prob = evaluation_result.split("[Probability]:")[-1].strip().split(' ')[0].strip().split('\n')[0].strip()
        return float(prob)
    else:
        return 0

def extract_label_conflict_detection(evaluation_result):
    lines = evaluation_result.strip().split('\n')
    for final_line in lines[::-1]:
        if final_line.lower().endswith("yes") or final_line.lower().endswith("no"):
            if final_line.lower().endswith("yes"):
                return "YES"
            else:
                return "NO"
        else:
            if "YES" in final_line and "NO" not in final_line:
                return "YES"
            elif "YES" not in final_line and "NO" in final_line:
                return "NO"
            else:
                # contain both NO and YES, or do not contain either
                pass
    return "No Answer"
        
def extract_label_instruction_following(evaluation_result):
    final_line = evaluation_result.strip().split('\n')[-1]
    if final_line.lower().endswith("yes") or final_line.lower().endswith("no"):
        if final_line.lower().endswith("yes"):
            return "YES"
        else:
            return "NO"
    else:
        if "YES" in final_line and "NO" not in final_line:
            return "YES"
        elif "YES" not in final_line and "NO" in final_line:
            return "NO"
        else:
            # contain both NO and YES, or do not contain either
            return "No Answer"
        
def extract_label(evaluation_result, instruction_order='new_org'):
    final_line = evaluation_result.strip().split('\n')[-1].strip()
    if final_line[-2:] == "-1":
        predict_label = -1
    elif final_line[-1]== "1":
        predict_label = 1
    elif final_line[-1] =="2":
        predict_label = 2
    else:
        print(final_line)
        predict_label = -1
        # raise ValueError()
    if instruction_order == "org_new":
        if predict_label ==1:
            predict_label =2
        elif predict_label ==2:
            predict_label=1
        else:
            pass 
    return  predict_label

def extract_label_conflict_resolution_behavior(evaluation_result):
    final_line = evaluation_result.strip().split('\n')[-1].strip()
    # print(final_line)
    # print(final_line[-1])
    if len(final_line)==0:
        print(evaluation_result)
        return -1
    if final_line[-1] == "1":
        predict_label = 1
    elif final_line[-1]== "2":
        predict_label = 2
    elif final_line[-1] =="3":
        predict_label = 3
    elif final_line[-1] =="4":
        predict_label = 4
    elif final_line[-1] =="5":
        predict_label = 5
    else:
        print(final_line)
        predict_label = -1
    return predict_label

def get_argument(parser):
    parser.add_argument('--max_tokens', type=int, default=2048, 
                        help='The number of max tokens for the generated output.')   
    parser.add_argument('--temperature', type=float, default=0)   
    parser.add_argument('--wait_time', type=int, default=10, 
                        help='Retry your request after a specified seconds.')  
    parser.add_argument('--retry_times', type=int, default=10, 
                        help='The maximum number of retry times.') 
    parser.add_argument('--task', type=str, default='conflict_resolution', 
                        choices=['instruction_following', 
                                 'conflict_resolution', 
                                 'conflict_detection', 
                                 'conflict_resolution_behavior', 'conflict_detection_prob'])
    parser.add_argument('--input_dir', type=str, default='./outputs/conflict_resolution/gpt-4o-2024-11-20/1') 
    parser.add_argument('--restart_idx', type=int, default=0)
    parser.add_argument('--instruction_type_idx', default="1-2-3-4-7", 
                        help='The type of constraints added to the original instruction.')
    args = parser.parse_args()
    args.instruction_type_idx_list = list_of_ints(args.instruction_type_idx)
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation.')
    parser.add_argument('--api_key', type=str, required=False, default='key') 
    parser.add_argument('--model', type=str, default="gpt-4o-2024-11-20", 
                        choices=['claude-sonnet-4-20250514', 'gemini-2.5-pro', 'deepseek-r1-250528',
                                 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 
                                 'gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18', 
                            ])  
    args = get_argument(parser)
    print(args)
    random.seed(42)
    output_dir = args.input_dir.replace("outputs", "evaluation_outputs")
    
    if args.task == 'conflict_resolution_behavior':
        if args.model=="claude-sonnet-4-20250514":
            output_dir = output_dir.replace("conflict_resolution_density", "conflict_resolution_density/claude-sonnet-4-20250514_behavior")
        elif args.model=='gemini-2.5-pro':
            output_dir = output_dir.replace("conflict_resolution_density", "conflict_resolution_density/gemini-2.5-pro_behavior")
        elif args.model=='deepseek-r1-250528':
            output_dir = output_dir.replace("conflict_resolution_density", "conflict_resolution_density/deepseek-r1-250528_behavior")
        else:
            assert args.model == "gpt-4o-2024-11-20"
            output_dir = output_dir.replace("conflict_resolution_density", "conflict_resolution_density/behavior")
            if "conflict_resolution_density" not in  output_dir:
                output_dir = output_dir.replace("conflict_resolution", "conflict_resolution/behavior")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    # exit()
    vllm = OpenAIGPT4(api_key=api_key_dict[args.api_key], model=args.model,
                      temperature=args.temperature, max_tokens=args.max_tokens,
                      wait_time=args.wait_time, retry_times=args.retry_times)
    if args.task == 'conflict_resolution':
        assert "conflict_resolution" in args.input_dir
        with open('./prompts/conflict_resolution_pair_evaluation.txt', 'r', encoding='utf-8') as fr:
            prompt_template = fr.read()
        for i, data_dict in read_input(args.input_dir):
            if i<=args.restart_idx:
                continue
            response_dict = {}
            prompt_tmp = prompt_template.replace("##model_response##", data_dict['response'])
            conflict_dict = data_dict['conflicts']
            for conflict_type in conflict_dict:
                response_dict[conflict_type] = {}
                new_constraint = conflict_dict[conflict_type]["new_constraint"]
                org_constraint = conflict_dict[conflict_type]["org_constraint"]
                if random.random()>0.5:
                    prompt = prompt_tmp.replace("##instruction1##", org_constraint).replace("##instruction2##", new_constraint)
                    response_dict[conflict_type]["instruction_order"] = "org_new"
                else:
                    prompt = prompt_tmp.replace("##instruction1##", new_constraint).replace("##instruction2##", org_constraint)
                    response_dict[conflict_type]["instruction_order"] = "new_org"
                response = vllm.generate(user_msg=prompt)
                response_dict[conflict_type]["new_constraint"] = new_constraint
                response_dict[conflict_type]["org_constraint"] = org_constraint
                response_dict[conflict_type]["evaluation_result"] = response
                # print(prompt)
            response_dict["llm_response"] = data_dict['response']
            output_filename = os.path.join(output_dir, f'{i}.json')
            with open(output_filename, 'w', encoding='utf-8') as fw:
                json.dump(response_dict, fw, indent = 4)
        
        # compute the success rate for org_constraint, new_constraint.
        success_rate_dict = {}
        avg_new_constraint_sr = 0
        avg_org_constraint_sr = 0
        total = 0
        for i in range(1, 101):
            with open(f"{output_dir}/{i}.json", 'r', encoding='utf-8') as fr:
                response_dict = json.load(fr)
                for conflict_type in response_dict:
                    if len(response_dict[conflict_type]) == 4:
                        if conflict_type not in success_rate_dict:
                            success_rate_dict[conflict_type] = {"new_constraint_sr": 0,  "org_constraint_sr":0, "num": 0}
                        print(i)
                        label = extract_label(response_dict[conflict_type]["evaluation_result"], response_dict[conflict_type]["instruction_order"])
                        if label == 1:
                            success_rate_dict[conflict_type]["new_constraint_sr"] += 1
                            avg_new_constraint_sr += 1
                        elif label ==2:
                            success_rate_dict[conflict_type]["org_constraint_sr"] += 1
                            avg_org_constraint_sr += 1
                        else:
                            pass
                        total +=1 
                        success_rate_dict[conflict_type]["num"] += 1
        for conflict_type in success_rate_dict:
            success_rate_dict[conflict_type]["new_constraint_sr"] /= success_rate_dict[conflict_type]["num"]
            success_rate_dict[conflict_type]["org_constraint_sr"] /= success_rate_dict[conflict_type]["num"]
        assert total == 100
        success_rate_dict["avg_new_constraint_sr"] = avg_new_constraint_sr/total
        success_rate_dict["avg_org_constraint_sr"] = avg_org_constraint_sr/total
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
            json.dump(success_rate_dict, fw, indent=4)
    elif args.task == 'conflict_resolution_behavior':
        assert "conflict_resolution" in args.input_dir
        if "0_conflict" in args.input_dir:
            with open('./prompts/instruction_following_behavior_evaluation.txt', 'r', encoding='utf-8') as fr:
                prompt_template = fr.read()
        else:
            with open('./prompts/conflict_resolution_behavior_evaluation.txt', 'r', encoding='utf-8') as fr:
                prompt_template = fr.read()
        for i, data_dict in read_input(args.input_dir):
            if i<=args.restart_idx or data_dict is None:
                continue
            prompt_tmp = prompt_template.replace("##model_response##", data_dict['response'])
            prompt = prompt_tmp.replace("##instruction##", data_dict['instruction'])
            response = vllm.generate(user_msg=prompt)
            response_dict = {}
            response_dict['instruction'] = data_dict['instruction']
            response_dict["llm_response"] = data_dict['response']
            if "conflicts" in data_dict:
                response_dict["conflicts"] = data_dict["conflicts"]
            response_dict["evaluation_result"] = response
            
            output_filename = os.path.join(output_dir, f'{i}.json')
            with open(output_filename, 'w', encoding='utf-8') as fw:
                json.dump(response_dict, fw, indent = 4)
            # exit()
        
        result_dict = {1:0, 2:0, 3:0, 4:0, -1:0,
                       "1_ids": [], "2_ids": [], "3_ids": [], "4_ids": [], "5_ids": []}
        for i in range(1, 101):
            if not os.path.exists(f"{output_dir}/{i}.json"):
                continue
            with open(f"{output_dir}/{i}.json", 'r', encoding='utf-8') as fr:
                response_dict = json.load(fr)
                print(i)
                label = extract_label_conflict_resolution_behavior(response_dict['evaluation_result'])
                # if i==17:
                #     exit()
                result_dict[label]+=1
                if label==-1:
                    result_dict["5_ids"].append(i)
                else:
                    result_dict[f"{label}_ids"].append(i)
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
            json.dump(result_dict, fw, indent=4)    
              
    elif args.task == 'instruction_following':
        assert "instruction_following" in args.input_dir
        with open('./prompts/instruction_following_evaluation.txt', 'r', encoding='utf-8') as fr:
            prompt_template = fr.read()
        with open('./datasets/conflict_instruction.jsonl', 'r', encoding='utf-8') as fr:
            conflict_data_list = fr.readlines()
        for i, data_dict in read_input(args.input_dir):
            if i<=args.restart_idx:
                continue
            # if i>25:
                # break
            response_dict = {}
            prompt_tmp = prompt_template.replace("##model_response##", data_dict['response'])
            # instruction_list
            conflict_data_dict = json.loads(conflict_data_list[i-1])
            instruction_2_response_dict = {}
            for instruction_type_idx in args.instruction_type_idx_list:
                instruction_type = idx_2_conflict_type[instruction_type_idx]
                instruction = conflict_data_dict[instruction_type]['org_constraint']
                
                if instruction in instruction_2_response_dict:
                    response = instruction_2_response_dict[instruction]
                else:
                    prompt = prompt_tmp.replace("##instruction##", instruction)
                    # print(prompt)
                    response = vllm.generate(user_msg=prompt)
                response_dict[instruction_type] = {}
                response_dict[instruction_type]["org_constraint"] = instruction
                response_dict[instruction_type]["evaluation_result"] = response
                instruction_2_response_dict[instruction] = response
                # print(prompt)
            response_dict["llm_response"] = data_dict['response']
            output_filename = os.path.join(output_dir, f'{i}.json')
            with open(output_filename, 'w', encoding='utf-8') as fw:
                json.dump(response_dict, fw, indent = 4)
                
        # compute the success rate for instructions.
        success_rate_dict = {}
        for i in range(1, 101):
            with open(f"{output_dir}/{i}.json", 'r', encoding='utf-8') as fr:
                response_dict = json.load(fr)
                for conflict_type in response_dict:
                    print(i)
                    if len(response_dict[conflict_type]) == 2:
                        if conflict_type not in success_rate_dict:
                            success_rate_dict[conflict_type] = {"org_constraint_sr":0}
                        label = extract_label_instruction_following(response_dict[conflict_type]["evaluation_result"])
                        if label == "YES":
                            success_rate_dict[conflict_type]["org_constraint_sr"] += 1/100
        avg_org_constraint_sr = 0
        for conflict_type in success_rate_dict:
            avg_org_constraint_sr += success_rate_dict[conflict_type]["org_constraint_sr"]/len(success_rate_dict)
        success_rate_dict["avg_org_constraint_sr"] = avg_org_constraint_sr
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
            json.dump(success_rate_dict, fw, indent=4)
    elif args.task == "conflict_detection":
        assert "conflict_detection" in args.input_dir
        no_dict = {"NO":0, "YES": 0, "No Answer": 0 }
        yes_dict = {"NO":0, "YES": 0, "No Answer": 0 }
        print(f"{os.path.dirname(args.input_dir)}/0".replace("conflict_detection_density",  "conflict_detection"))
        for i, data_dict in read_input(f"{os.path.dirname(args.input_dir)}/0".replace("conflict_detection_density",  "conflict_detection")):
            if data_dict is None:
                continue
            label = extract_label_conflict_detection(data_dict['response'])
            no_dict[label] += 1
                
        for i, data_dict in read_input(args.input_dir):
            if data_dict is None:
                continue
            label = extract_label_conflict_detection(data_dict['response'])
            yes_dict[label] += 1
        # assert sum(no_dict.values()) == 100
        # assert sum(yes_dict.values()) == 100
        print(no_dict)
        print(yes_dict)
        accuracy = (no_dict['NO']+yes_dict["YES"])/(sum(no_dict.values())+sum(yes_dict.values()))
        precision  = yes_dict["YES"]/(yes_dict["YES"]+no_dict['YES'])
        recall = yes_dict["YES"]/sum(yes_dict.values())
        F1 = 2*precision*recall/(precision+recall)
        result_dict = {
            "no_predicted_result": no_dict,
            "yes_predicted_result": yes_dict,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": F1
        }
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
            json.dump(result_dict, fw, indent=4)
    elif args.task == 'conflict_detection_prob':
        assert "conflict_detection_prob" in args.input_dir
        predicted_prob_list = []
        for i, data_dict in read_input(args.input_dir):
            if data_dict is None:
                continue
            prob = extract_conflict_detection_prob(data_dict['response'])
            predicted_prob_list.append(prob)
        if '0_conflict' in args.input_dir:
            gold_prob_list = [0]*len(predicted_prob_list)
        else:
            gold_prob_list = [1]*len(predicted_prob_list)
        result_dict = {
            "predicted_prob_list": predicted_prob_list,
            "gold_prob_list": gold_prob_list,
            "num_samples": len(predicted_prob_list)
        }
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
            json.dump(result_dict, fw, indent=4)
    else:
        pass
        
        
        
