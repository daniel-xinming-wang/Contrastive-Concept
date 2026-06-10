model=gpt-4o-2024-11-20
for num_conflict in 1 2 3 4 5 6
do
    python LLMs/gpt4.py --model $model --task conflict_resolution_density --num_conflict $num_conflict
    input_dir=./outputs/conflict_resolution_density/$model/${num_conflict}_conflict 
    python LLMs/evaluation.py --input_dir $input_dir --task conflict_resolution_behavior --model gpt-4o-2024-11-20
done