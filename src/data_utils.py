import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def load_config(config_path: str) -> dict:
    """Load YAML config file. Every script calls this — single source of truth."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_sample(sample: dict) -> str:
    """
    Convert a raw Dolly sample into a structured prompt string.

    Dolly samples have three fields:
      - instruction: what the model should do
      - context: optional background text (e.g. a paragraph to summarize)
      - response: the correct answer

    We format these into the ChatML template that instruct models expect.
    This exact format is what was used during the base model's instruction tuning,
    so staying consistent with it helps the model understand the task faster.
    """
    instruction = sample.get("instruction", "").strip()
    context = sample.get("context", "").strip()
    response = sample.get("response", "").strip()

    if context:
        user_message = f"{instruction}\n\n{context}"
    else:
        user_message = instruction
    prompt = (
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )

    return prompt


def tokenize_sample(sample: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    """
    Tokenize a single formatted prompt and build labels with prompt masking.

    Key concept: we only want the model to learn from the RESPONSE tokens,
    not the instruction tokens. So we mask the instruction part of the labels
    with -100, which CrossEntropyLoss ignores.

    This is the same fix we applied in the receipt project — now done cleanly
    as a reusable function.
    """
    full_prompt = format_sample(sample)

    # Tokenize the full prompt (instruction + response)
    tokenized = tokenizer(
        full_prompt,
        max_length=max_length,
        truncation=True,
        padding=False,       # we pad later in the collator, not here
        return_tensors=None, # return plain lists, not tensors (datasets library prefers this)
    )

    labels = tokenized["input_ids"].copy()

    instruction_part = (
        f"<|im_start|>user\n{sample.get('instruction', '').strip()}"
    )
    if sample.get("context", "").strip():
        instruction_part += f"\n\n{sample.get('context', '').strip()}"
    instruction_part += f"<|im_end|>\n<|im_start|>assistant\n"

    instruction_tokens = tokenizer(
        instruction_part,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    prompt_len = len(instruction_tokens["input_ids"])

    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


def get_dataset(config_path: str, tokenizer: AutoTokenizer):
    """
    Load, format, tokenize, and split the Dolly dataset.
    Returns a DatasetDict with 'train' and 'validation' splits.
    """
    config = load_config(config_path)
    data_cfg = config["data"]

    print(f"[data] Loading {data_cfg['dataset_name']}...")
    dataset = load_dataset(data_cfg["dataset_name"], split="train")

    dataset = dataset.shuffle(seed=42).select(range(data_cfg["num_samples"]))
    print(f"[data] Using {len(dataset)} samples")

    print("[data] Tokenizing...")
    dataset = dataset.map(
        lambda sample: tokenize_sample(sample, tokenizer, data_cfg["max_length"]),
        remove_columns=dataset.column_names,
    )

    # Train / validation split
    split = dataset.train_test_split(
        test_size=1 - data_cfg["train_split"],
        seed=42,
    )

    print(f"[data] Train: {len(split['train'])} samples | Val: {len(split['test'])} samples")
    return split["train"], split["test"]