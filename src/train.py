import os
import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from data_utils import load_config, get_dataset


def get_device():
    """
    Detect the best available device in order of preference:
    CUDA (Nvidia) > MPS (Apple Silicon) > CPU.
    Returns both the device string and capability flags for TrainingArguments.
    """
    if torch.cuda.is_available():
        return "cuda", True, False
    elif torch.backends.mps.is_available():
        return "mps", False, True
    else:
        return "cpu", False, False


def load_model_and_tokenizer(config: dict, device: str):
    """Load the base model and tokenizer onto the detected device."""
    model_name = config["model"]["name"]
    dtype = getattr(torch, config["model"]["torch_dtype"])

    print(f"[model] Loading {model_name} on {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},  # explicitly map all layers to our detected device
    )

    return model, tokenizer


def apply_lora(model, config: dict):
    """Wrap the base model with LoRA adapters — freezes base weights, injects A+B matrices."""
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def get_training_args(config: dict, use_cuda: bool, use_mps: bool) -> TrainingArguments:
    """
    Build TrainingArguments with device-aware flags.

    bf16=True only works on CUDA — passing it on MPS causes a crash.
    On MPS we rely on bfloat16 being set at model load time via torch_dtype,
    not via the Trainer flag.
    """
    train_cfg = config["training"]

    return TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=50,
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=train_cfg["save_best_only"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_cuda,           # CUDA only — bfloat16 training precision flag
        report_to="none",
        remove_unused_columns=False,
    )


def train():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "lora_config.yaml")
    config = load_config(config_path)

    # ── Step 1: Detect device ──────────────────────────────────────────────────
    device, use_cuda, use_mps = get_device()
    print(f"[device] Using: {device.upper()}")

    # ── Step 2: Load model and tokenizer ──────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config, device)

    # ── Step 3: Inject LoRA adapters, freeze base weights ─────────────────────
    model = apply_lora(model, config)

    # ── Step 4: Load and tokenize dataset ─────────────────────────────────────
    train_dataset, val_dataset = get_dataset(config_path, tokenizer)

    # ── Step 5: Data collator ──────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── Step 6: Build trainer and run ─────────────────────────────────────────
    training_args = get_training_args(config, use_cuda, use_mps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\n[train] Starting training...\n")
    trainer.train()

    # ── Step 7: Save the LoRA adapter (not the full model — adapter is ~30MB) ──
    adapter_path = config["training"]["output_dir"]
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n[train] LoRA adapter saved to {adapter_path}")

    # ── Step 8: Free memory ────────────────────────────────────────────────────
    del model
    gc.collect()
    if use_mps:
        torch.mps.empty_cache()


if __name__ == "__main__":
    train()