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