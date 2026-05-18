import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


DEFAULT_INPUT_DIRS = [
    "outputs/generations_vllm_0.3_base_to_neg_test_dosample0",
    "outputs/generations_vllm_0.3_base_to_pos_test_dosample0",
]

PROMPT_FILES = {
    "semantic_framing": "semantic_framing_eval_v1.txt",
    "ideology": "ideology_eval_v1.txt",
    "linguistic_style": "linguistic_style_eval_v1.txt",
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt_templates(prompt_dir):
    prompt_dir = Path(prompt_dir)
    templates = {}
    for category_key, filename in PROMPT_FILES.items():
        path = prompt_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            templates[category_key] = f.read()
    return templates


def find_generation_files(input_dirs):
    files = []
    for input_dir in input_dirs:
        files.extend(Path(input_dir).glob("**/generations.jsonl"))
    return sorted(files)


def infer_direction(path):
    path_str = str(path)
    if "base_to_pos" in path_str:
        return "base_to_pos"
    if "base_to_neg" in path_str:
        return "base_to_neg"
    raise ValueError(f"Could not infer direction from path: {path}")


def target_concept_for(row, direction):
    if direction == "base_to_pos":
        return row["positive_concept"]
    if direction == "base_to_neg":
        return row["negative_concept"]
    raise ValueError(f"Invalid direction: {direction}")


def build_eval_prompt(template, row, target_concept):
    return template.format(
        target_concept=target_concept,
        parsed_response=row["generated_text"].strip(),
    )


def parse_score(judgement):
    match = re.search(r"Score\s*(?:\([^)]+\))?\s*:\s*([01])", judgement, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback for terse outputs like "1\nExplanation: ..."
    match = re.search(r"^\s*([01])\b", judgement)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not parse Score from judgement: {judgement!r}")


class OpenAIJudge:
    def __init__(self, model_name):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024),
    )
    def get_judgement(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=96,
            temperature=0,
        )
        return response.choices[0].message.content


class LocalJudge:
    def __init__(self, model_name, max_new_tokens=96):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        ).eval()

    def get_judgement(self, prompt):
        chat = [
            {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
            {"role": "user", "content": prompt},
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        new_tokens = output[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class Gemma3Judge:
    def __init__(self, model_name, max_new_tokens=96):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        ).eval()

    def get_judgement(self, prompt):
        chat = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant who follows instructions exactly."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = output[0][prompt_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


def make_record(row, generation_file, direction, target_concept, judgement, score, judge_model):
    return {
        "judge_model": judge_model,
        "generation_file": str(generation_file),
        "direction": direction,
        "target_concept": target_concept,
        "score": score,
        "judgement": judgement,
        "category_name": row.get("category_name"),
        "category_key": row["category_key"],
        "pair_slug": row["pair_slug"],
        "positive_concept": row["positive_concept"],
        "negative_concept": row["negative_concept"],
        "statement_index": row.get("statement_index"),
        "statement_source": row.get("statement_source"),
        "statement": row.get("statement"),
        "variant": row.get("variant"),
        "generated_text": row["generated_text"],
    }


def load_completed_keys(path):
    if not path.exists():
        return set()

    completed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(record_key(row))
    return completed


def record_key(row):
    return (
        row["generation_file"],
        row["direction"],
        row["pair_slug"],
        row.get("variant"),
        row.get("statement_index"),
        row["target_concept"],
    )


def candidate_key(row, generation_file, direction, target_concept):
    return (
        str(generation_file),
        direction,
        row["pair_slug"],
        row.get("variant"),
        row.get("statement_index"),
        target_concept,
    )


def compute_metrics(records):
    groups = defaultdict(list)
    for row in records:
        keys = {
            "overall": ("overall",),
            "by_direction": (row["direction"],),
            "by_category": (row["category_key"],),
            "by_pair": (row["category_key"], row["pair_slug"]),
            "by_pair_direction": (row["category_key"], row["pair_slug"], row["direction"]),
            "by_target": (row["category_key"], row["pair_slug"], row["direction"], row["target_concept"]),
        }
        for group_name, key in keys.items():
            groups[(group_name, key)].append(row["score"])

    metrics = {}
    for (group_name, key), scores in groups.items():
        metrics.setdefault(group_name, [])
        metrics[group_name].append(
            {
                "key": list(key),
                "n": len(scores),
                "success_rate": sum(scores) / len(scores) if scores else 0.0,
                "num_success": int(sum(scores)),
            }
        )

    for values in metrics.values():
        values.sort(key=lambda x: x["key"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", default=DEFAULT_INPUT_DIRS)
    parser.add_argument("--prompt_dir", default="contrastive_hidden_states/data/evaluation_prompts")
    parser.add_argument("--output_dir", default="outputs/judgements_vllm_0.3_dosample0")
    parser.add_argument("--judge_type", choices=["openai", "local", "llama", "gemma3"], default="openai")
    parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--category_key", default=None)
    parser.add_argument("--pair_slug", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    for name, value in args.__dict__.items():
        print(f"{name:<20} : {value}")

    templates = load_prompt_templates(args.prompt_dir)
    generation_files = find_generation_files(args.input_dirs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgements_path = output_dir / "judgements.jsonl"
    metrics_json_path = output_dir / "metrics.json"
    metrics_pkl_path = output_dir / "metrics.pkl"

    completed = set() if args.no_resume else load_completed_keys(judgements_path)
    if args.dry_run:
        judge = None
    elif args.judge_type == "openai":
        judge = OpenAIJudge(args.judge_model)
    elif args.judge_type in ["local", "llama"]:
        judge = LocalJudge(args.judge_model)
    elif args.judge_type == "gemma3":
        judge = Gemma3Judge(args.judge_model)
    else:
        raise ValueError(f"Invalid judge type: {args.judge_type}")

    written = []
    processed = 0

    for generation_file in generation_files:
        direction = infer_direction(generation_file)
        rows = read_jsonl(generation_file)
        for row in tqdm(rows, desc=str(generation_file)):
            if args.category_key and row["category_key"] != args.category_key:
                continue
            if args.pair_slug and row["pair_slug"] != args.pair_slug:
                continue
            if row["category_key"] not in templates:
                continue

            target_concept = target_concept_for(row, direction)
            key = candidate_key(row, generation_file, direction, target_concept)
            if key in completed:
                continue

            prompt = build_eval_prompt(templates[row["category_key"]], row, target_concept)

            if args.dry_run:
                print("=" * 100)
                print(prompt)
                processed += 1
                if args.limit is not None and processed >= args.limit:
                    return
                continue

            judgement = judge.get_judgement(prompt)
            score = parse_score(judgement)
            record = make_record(
                row=row,
                generation_file=generation_file,
                direction=direction,
                target_concept=target_concept,
                judgement=judgement,
                score=score,
                judge_model=args.judge_model,
            )
            append_jsonl(judgements_path, [record])
            written.append(record)
            processed += 1

            if args.limit is not None and processed >= args.limit:
                break

        if args.limit is not None and processed >= args.limit:
            break

    all_records = []
    if judgements_path.exists():
        all_records = read_jsonl(judgements_path)

    metrics = compute_metrics(all_records)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(metrics_pkl_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"New judgements: {len(written)}")
    print(f"Total judgements: {len(all_records)}")
    print(f"Saved judgements to {judgements_path}")
    print(f"Saved metrics to {metrics_json_path}")
    print(f"Saved metrics pickle to {metrics_pkl_path}")


if __name__ == "__main__":
    main()
