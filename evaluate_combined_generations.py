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
    "outputs/generations_negpos_full",
    "outputs/generations_posneg_full",
]

DEFAULT_PROMPT_FILE = "contrastive_hidden_states/data/evaluation_prompts/combined_concepts_success_eval_v1.txt"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_generation_files(input_dirs):
    files = []
    for input_dir in input_dirs:
        files.extend(Path(input_dir).glob("**/generations.jsonl"))
    return sorted(files)


def generation_model_matches(path, generation_model):
    if generation_model is None:
        return True

    path_parts = set(Path(path).parts)
    requested = {generation_model, Path(generation_model).name}
    return bool(path_parts & requested)


def infer_combined_order(path, row=None):
    parts = Path(path).parts
    if "negpos" in parts:
        return "negpos"
    if "posneg" in parts:
        return "posneg"

    if row is not None and row.get("variant") in {"negpos", "posneg"}:
        return row["variant"]

    raise ValueError(f"Could not infer combined order from path: {path}")


def ordered_concepts(row, combined_order):
    if combined_order == "negpos":
        return row["negative_concept"], row["positive_concept"]
    if combined_order == "posneg":
        return row["positive_concept"], row["negative_concept"]
    raise ValueError(f"Invalid combined order: {combined_order}")


def build_eval_prompt(template, row, first_concept, second_concept):
    return template.format(
        first_concept=first_concept,
        second_concept=second_concept,
        category_key=row["category_key"],
        statement=(row.get("statement") or "").strip(),
        generated_text=row["generated_text"].strip(),
    )


def parse_success(judgement):
    match = re.search(r"Success\s*(?:\([^)]+\))?\s*:\s*([01])", judgement, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"Score\s*(?:\([^)]+\))?\s*:\s*([01])", judgement, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"^\s*([01])\b", judgement)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not parse Success from judgement: {judgement!r}")


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
                {"role": "system", "content": "You are a careful evaluator who follows the requested output format exactly."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=128,
            temperature=0,
        )
        return response.choices[0].message.content


class LocalJudge:
    def __init__(self, model_name, max_new_tokens=128):
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
            {"role": "system", "content": "You are a careful evaluator who follows the requested output format exactly."},
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


class VLLMJudge:
    def __init__(
        self,
        model_name,
        max_new_tokens=128,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=None,
        trust_remote_code=False,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            legacy=False,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

    def wrap_prompt(self, prompt):
        chat = [
            {"role": "system", "content": "You are a careful evaluator who follows the requested output format exactly."},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_judgements(self, prompts):
        wrapped_prompts = [self.wrap_prompt(prompt) for prompt in prompts]
        outputs = self.llm.generate(
            wrapped_prompts,
            self.sampling_params,
            use_tqdm=False,
        )
        return [output.outputs[0].text.strip() for output in outputs]

    def get_judgement(self, prompt):
        return self.get_judgements([prompt])[0]


class Gemma3Judge:
    def __init__(self, model_name, max_new_tokens=128):
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
            {"role": "system", "content": [{"type": "text", "text": "You are a careful evaluator who follows the requested output format exactly."}]},
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


def make_record(row, generation_file, combined_order, first_concept, second_concept, judgement, success, judge_model):
    return {
        "judge_model": judge_model,
        "generation_file": str(generation_file),
        "combined_order": combined_order,
        "first_concept": first_concept,
        "second_concept": second_concept,
        "success": success,
        "score": success,
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


def record_key(row):
    return (
        row["generation_file"],
        row["combined_order"],
        row["pair_slug"],
        row.get("variant"),
        row.get("statement_index"),
        row["first_concept"],
        row["second_concept"],
    )


def candidate_key(row, generation_file, combined_order, first_concept, second_concept):
    return (
        str(generation_file),
        combined_order,
        row["pair_slug"],
        row.get("variant"),
        row.get("statement_index"),
        first_concept,
        second_concept,
    )


def load_completed_keys(path):
    if not path.exists():
        return set()

    completed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            completed.add(record_key(json.loads(line)))
    return completed


def compute_metrics(records):
    groups = defaultdict(list)
    for row in records:
        keys = {
            "overall": ("overall",),
            "by_order": (row["combined_order"],),
            "by_category": (row["category_key"],),
            "by_category_order": (row["category_key"], row["combined_order"]),
            "by_pair": (row["category_key"], row["pair_slug"]),
            "by_pair_order": (row["category_key"], row["pair_slug"], row["combined_order"]),
        }
        for group_name, key in keys.items():
            groups[(group_name, key)].append(row["success"])

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
    parser.add_argument("--prompt_file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output_dir", default="outputs/judgements_combined_full")
    parser.add_argument("--judge_type", choices=["openai", "local", "llama", "gemma3", "vllm"], default="openai")
    parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--judge_max_new_tokens", type=int, default=128)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--judge_batch_size", type=int, default=16)
    parser.add_argument("--category_key", default=None)
    parser.add_argument("--pair_slug", default=None)
    parser.add_argument("--generation_model", default=None)
    parser.add_argument("--combined_order", choices=["negpos", "posneg"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    for name, value in args.__dict__.items():
        print(f"{name:<20} : {value}")

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        template = f.read()

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
    elif args.judge_type == "vllm":
        judge = VLLMJudge(
            model_name=args.judge_model,
            max_new_tokens=args.judge_max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(f"Invalid judge type: {args.judge_type}")

    written = []
    processed = 0
    pending = []

    def flush_pending():
        nonlocal processed

        if not pending:
            return

        prompts = [item["prompt"] for item in pending]
        if hasattr(judge, "get_judgements"):
            judgements = judge.get_judgements(prompts)
        else:
            judgements = [judge.get_judgement(prompt) for prompt in prompts]

        records = []
        for item, judgement in zip(pending, judgements):
            success = parse_success(judgement)
            records.append(
                make_record(
                    row=item["row"],
                    generation_file=item["generation_file"],
                    combined_order=item["combined_order"],
                    first_concept=item["first_concept"],
                    second_concept=item["second_concept"],
                    judgement=judgement,
                    success=success,
                    judge_model=args.judge_model,
                )
            )

        append_jsonl(judgements_path, records)
        written.extend(records)
        processed += len(records)
        pending.clear()

    for generation_file in generation_files:
        if not generation_model_matches(generation_file, args.generation_model):
            continue

        file_combined_order = infer_combined_order(generation_file)
        if args.combined_order and file_combined_order != args.combined_order:
            continue

        rows = read_jsonl(generation_file)
        for row in tqdm(rows, desc=str(generation_file)):
            if args.category_key and row["category_key"] != args.category_key:
                continue
            if args.pair_slug and row["pair_slug"] != args.pair_slug:
                continue

            combined_order = file_combined_order
            first_concept, second_concept = ordered_concepts(row, combined_order)
            key = candidate_key(row, generation_file, combined_order, first_concept, second_concept)
            if key in completed:
                continue

            prompt = build_eval_prompt(template, row, first_concept, second_concept)

            if args.dry_run:
                print("=" * 100)
                print(prompt)
                processed += 1
                if args.limit is not None and processed >= args.limit:
                    return
                continue

            pending.append(
                {
                    "row": row,
                    "generation_file": generation_file,
                    "combined_order": combined_order,
                    "first_concept": first_concept,
                    "second_concept": second_concept,
                    "prompt": prompt,
                }
            )

            should_flush = len(pending) >= args.judge_batch_size
            if args.limit is not None and processed + len(pending) >= args.limit:
                should_flush = True
            if should_flush:
                flush_pending()

            if args.limit is not None and processed >= args.limit:
                break

        flush_pending()

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
