# reimplement the conflict detection results in Table 2

# for proprietary LLMs
model=gpt-4o-mini-2024-07-18
for conflict_type_idx in 0
do 
    python LLMs/gpt4.py --model $model --conflict_type_idx $conflict_type_idx --task conflict_detection --num_conflict 1
done

for conflict_type_idx in 1 2 3 4 5 6 7 8 9
do 
    python LLMs/evaluation.py --input_dir ./outputs/conflict_detection/${model}/$conflict_type_idx --task conflict_detection
done


# for open-source LLMs
model=Qwen/Qwen2.5-3B-Instruct
for conflict_type_idx in 0 1 2 3 4 5 6 7 8 9
do 
    python LLMs/llama.py --model $model --conflict_type_idx $conflict_type_idx --task conflict_detection --num_conflict 1
done

for conflict_type_idx in 1 2 3 4 5 6 7 8 9
do 
    python LLMs/evaluation.py --input_dir ./outputs/conflict_detection/${model}/$conflict_type_idx --task conflict_detection
done