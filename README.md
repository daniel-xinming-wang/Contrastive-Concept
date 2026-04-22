# Contrastive Hidden States

The goal is to:

1. Parse the contrastive concept pairs.
2. Build `negative / base / positive` prompts from three local statement classes.
3. Run a local Hugging Face causal LM with `output_hidden_states=True`.
4. Save hidden states separately for each prompt variant.
5. Optionally run text generation for the same prompt set and save responses separately.

## Current Logic

### 1. Concept parsing

The file [contrastive_hidden_states/concepts.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/concepts.py) reads `contrastive_concepts.txt`.

It treats lines without `vs` as category headers, for example:

- `LINGUISTIC STYLE`
- `IDEOLOGY`
- `SEMANTIC FRAMING`

It treats lines with `vs` as one contrastive pair, for example:

- `concise vs verbose`
- `risk-averse vs risk-seeking`

Each pair is stored as a `ContrastivePair` with:

- `category_name`
- `category_key`
- `positive`
- `negative`
- `slug`

The current parser assumes the left side of `A vs B` is the `positive` concept and the right side is the `negative` concept for bookkeeping. This does not yet imply any learned steering direction.

### 2. Statement files

The current pipeline no longer reads the upstream `400_general_statements` files directly at runtime.

Instead, it reads our own three statement files:

- [class_0.txt](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/data/statements_300/class_0.txt)
- [class_1.txt](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/data/statements_300/class_1.txt)
- [class_2.txt](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/data/statements_300/class_2.txt)

These were created by taking the first `300` statements from the original upstream statement pool and splitting them sequentially into three groups of exactly `100`.

The mapping is:

- `class_0.txt -> negative`
- `class_1.txt -> base`
- `class_2.txt -> positive`

So the three prompt variants do not share the same statement text. Each variant reads from its own class file.

This is the current default runtime behavior in [extract_hidden_states.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/extract_hidden_states.py):

- `negative` prompts are built only from `class_0.txt`
- `base` prompts are built only from `class_1.txt`
- `positive` prompts are built only from `class_2.txt`

The generation pipeline in [generate_responses.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/generate_responses.py) uses the same mapping:

- `negative` generation prompts use only `class_0.txt`
- `base` generation prompts use only `class_1.txt`
- `positive` generation prompts use only `class_2.txt`

### 3. Prompt construction

The file [contrastive_hidden_states/prompts.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/prompts.py) builds prompts for each concept pair using three separate statement groups.

The three variants are:

- `negative`
- `base`
- `positive`

The base template is currently:

```text
What are your thoughts on the following statement?
Statement: {statement}
Response:
```

Then category-specific instructions are prepended:

- `linguistic_style`: `Adopt the following linguistic style: {concept}.`
- `ideology`: `Adopt the following ideological stance: {concept}.`
- `semantic_framing`: `Adopt the following semantic framing: {concept}.`

For example, for `concise vs verbose`, the three prompts are built from three different statement files:

```text
negative:
Adopt the following linguistic style: verbose.

What are your thoughts on the following statement?
Statement: <statement from class_0.txt>
Response:

base:
What are your thoughts on the following statement?
Statement: <statement from class_1.txt>
Response:

positive:
Adopt the following linguistic style: concise.

What are your thoughts on the following statement?
Statement: <statement from class_2.txt>
Response:
```

If the tokenizer supports chat templates, the prompt is wrapped with `tokenizer.apply_chat_template(...)` before extraction.

This means the current code is not using a paired-triplet setup where the same statement is reused for all three variants. It is using three distinct statement pools.

### 4. Model loading

The file [contrastive_hidden_states/models.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/models.py) loads a local Hugging Face model and tokenizer.

This follows the same overall style as `neural_controllers-xrfm`:

- use `AutoModelForCausalLM.from_pretrained(...)`
- use `AutoTokenizer.from_pretrained(...)`
- infer the default hidden layers from `model.config.num_hidden_layers`

You can pass either a model alias configured in the codebase, a raw Hugging Face model id, or a local model path. Actual availability depends on the runtime environment and model access permissions.

### 5. Hidden-state extraction

The file [contrastive_hidden_states/hidden_states.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/hidden_states.py) does the actual extraction.

The key behavior is:

- tokenize prompts with padding
- run the model with `output_hidden_states=True`
- collect hidden states from all requested layers
- by default, take the last token representation with `rep_token = -1`

So for each prompt, the saved representation for a layer is:

- one vector per prompt
- corresponding to the final token hidden state at that layer

This matches the default extraction style used in the upstream `neural_controllers-xrfm` code.

### 6. Saving layout

The CLI currently saves each concept pair independently.

The entry point is [extract_hidden_states.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/extract_hidden_states.py).

For each pair, outputs are written under:

```text
outputs/hidden_states/<model>/<category>/<pair_slug>/
```

Inside each pair directory:

```text
metadata.json
negative/
  layer_-1.npy
  layer_-2.npy
  ...
base/
  layer_-1.npy
  layer_-2.npy
  ...
positive/
  layer_-1.npy
  layer_-2.npy
  ...
```

Notes:

- `negative`, `base`, and `positive` are saved separately.
- Default format is `.npy`.
- You can switch to `.pt` with `--save-format pt`.
- `metadata.json` stores run settings, example-level prompt metadata, and saved file names.

For `.npy` mode, each file contains one layer for one variant, shaped roughly as:

```text
(num_statements, hidden_dim)
```

If `--max-statements 20`, each saved layer file for one variant will have 20 rows from its own class file.

In other words:

- `--max-statements 20` means 20 `negative` statements, 20 `base` statements, and 20 `positive` statements
- total prompts per concept pair will then be 60

## File Overview

- [extract_hidden_states.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/extract_hidden_states.py): CLI entry point
- [generate_responses.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/generate_responses.py): CLI entry point for text generation
- [contrastive_hidden_states/concepts.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/concepts.py): parse categories and contrastive pairs
- [contrastive_hidden_states/prompts.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/prompts.py): build `negative / base / positive` prompts
- [contrastive_hidden_states/models.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/models.py): load local HF model and tokenizer
- [contrastive_hidden_states/hidden_states.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/hidden_states.py): extract and save hidden states
- [contrastive_hidden_states/generation.py](/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Wisc/contrastive_hidden_states/generation.py): generate and save text outputs

## How To Run

### Small test run

This is the recommended first run:

```bash
python extract_hidden_states.py \
  --model llama_3_8b_it \
  --categories linguistic_style \
  --max-pairs-per-category 1 \
  --max-statements 5 \
  --batch-size 2 \
  --save-format npy
```

This run uses:

- 5 rows from `class_0.txt` for `negative`
- 5 rows from `class_1.txt` for `base`
- 5 rows from `class_2.txt` for `positive`

### Generation run

To generate text outputs instead of hidden states, run:

```bash
python generate_responses.py \
  --model qwen2_5_7b_it \
  --max-pairs-per-category 5 \
  --max-statements 10 \
  --batch-size 2 \
  --max-new-tokens 128
```

This run uses the same statement mapping as above:

- `negative -> class_0.txt`
- `base -> class_1.txt`
- `positive -> class_2.txt`

By default, this will process:

- first `5` pairs from `linguistic_style`
- first `5` pairs from `ideology`
- first `5` pairs from `semantic_framing`
- first `10` statements from each statement class

Outputs are saved under:

```text
outputs/generations/<model>/<category>/<pair_slug>/
```

Each pair directory contains:

```text
metadata.json
negative/generations.jsonl
base/generations.jsonl
positive/generations.jsonl
```

### Larger run

```bash
python extract_hidden_states.py \
  --model llama_3_8b_it \
  --max-statements 100 \
  --batch-size 4
```

### Save as PyTorch instead of NumPy

```bash
python extract_hidden_states.py \
  --model llama_3_8b_it \
  --save-format pt
```

## What This Pipeline Does Not Do Yet

This code currently does not:

- compute steering vectors
- train RFM or linear probes
- compute `positive - base` or `positive - negative` differences
- reuse the same statement across all three prompt variants
- aggregate concept pairs into category-level vectors

It is only the hidden-state extraction stage.

## Recommended Next Step

The next clean step is to add a loader that reads:

- `negative`
- `base`
- `positive`

back into aligned per-layer dictionaries, so downstream code can directly compute:

- `positive - base`
- `negative - base`
- `positive - negative`

without reparsing `metadata.json` manually.
