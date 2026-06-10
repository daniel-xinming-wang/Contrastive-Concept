# ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions

Instruction-following is a critical capability of Large Language Models (LLMs). While existing works primarily focus on assessing how well LLMs adhere to user instructions, they often overlook scenarios where instructions contain conflicting constraints—a common occurrence in complex prompts. The behavior of LLMs under such conditions remains under-explored. To bridge this gap, we introduce ConInstruct, a benchmark specifically designed to assess LLMs’ ability to detect and resolve conflicts within user instructions. Using this dataset, we evaluate LLMs’ conflict detection performance and analyze their conflict resolution behavior. Our experiments reveal two key findings: (1) Most proprietary LLMs exhibit strong conflict detection capabilities, whereas among open-source models, only DeepSeek-R1 demonstrates similarly strong performance. DeepSeek-R1 and Claude-4.5-Sonnet achieve the highest average F1-scores at 91.5% and 87.3%, respectively, ranking first and second overall. (2) Despite their strong conflict detection abilities, LLMs rarely explicitly notify users about the conflicts or request clarification when faced with conflicting constraints. These results underscore a critical shortcoming in current LLMs and highlight an important area for future improvement when designing instruction-following LLMs. 

## Requirements

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   
pip install bitsandbytes   
pip install accelerate transformers   
pip install nltk   

You should set your API KEY in the following files before running the scripts:  
- line 3 of LLMs/evaluation.py  
- line 5 of LLMs/gpt4.py

## ConInstruct Dataset

We begin by manually curating 100 seed instructions, which serve as fundamental instructions without additional constraints. We then use GPT-4 to incorporate six types of constraints into each seed instruction, thereby generating a set of expanded instructions. Finally, GPT-4 is employed to introduce conflicting constraints into the expanded instructions, resulting in the ConInstruct dataset.

| Instruction Type | Dataset Path |
|------|----------|
| seed instructions | `datasets/seed_instruction.jsonl` |
| expanded instructions | `datasets/expand_instruction.jsonl` |
| conflicting instructions | `datasets/conflict_instruction.jsonl` |

ConInstruct is also available at **Hugging Face**: https://huggingface.co/datasets/He-Xingwei/ConInstruct. For specific data format details, please refer to the Hugging Face link above.

## Conflict Detection

Run conflict detection on LLMs with instructions containing different types of conflicts (Results in Table 2):

```bash
sh scripts/conflict_detection.sh
```

Run conflict detection on LLMs with instructions containing different numbers of conflicts (Results in Figure 3):

```bash
sh scripts/conflict_detection_density.sh
```

## Conflict Resolution

Run the following script to get the distributions of conflict resolution behaviors exhibited by LLMs:

```bash
sh scripts/conflict_resolution_density.sh
```

## Citation
If you want to use this code or dataset in your research, please cite our [paper](https://arxiv.org/abs/2511.14342):

```bash
@inproceedings{he2026ConInstruct,
  title={ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions},
  author={Xingwei He, and Qianru Zhang, and Pengfei Chen, and Guanhua Chen, and Linlin Yu, and Yuan Yuan, and Siu-Ming Yiu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
