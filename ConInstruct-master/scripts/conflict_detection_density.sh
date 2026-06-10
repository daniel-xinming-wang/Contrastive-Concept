model=Qwen/Qwen2.5-3B-Instruct
python LLMs/llama.py --model $model \
            --task conflict_detection_density \
            --num_conflict 2
               
input_dir=./outputs/conflict_detection_density/$model/2_conflict 
python LLMs/evaluation.py --input_dir $input_dir --task conflict_detection



