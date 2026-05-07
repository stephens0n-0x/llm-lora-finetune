import os
import gc
import yaml
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


def load_model_and_tokenizer(config: dict):
    """
    Load the base model and tokenizer.

    We load in bfloat16 to halve memory usage vs float32.
    We do NOT use 4-bit quantization (bitsandbytes) here because
    bitsandbytes has limited MPS support on Apple Silicon.
    QLoRA is the move when you have a CUDA GPU — on M1 we use
    bfloat16 + LoRA which is still very memory efficient.
    """
    model_name = config["model"]["name"]
    dtype = getattr(torch, config["model"]["torch_dtype"])  # converts "bfloat16" string to torch.bfloat16

    print(f"[model] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # SmolLM2 and Llama both need a padding token set explicitly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        # device_map="auto" lets accelerate decide whether to use MPS or CPU
        # it will use MPS for the model weights and fall back to CPU if needed
        device_map="auto",
    )

    return model, tokenizer


def apply_lora(model, config: dict):
    """
    Wrap the base model with LoRA adapters using the PEFT library.

    After this function, only the A and B matrices (the adapters) are
    trainable. The base model weights are completely frozen.
    This is the core of what makes LoRA memory efficient.
    """
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


def get_training_args(config: dict) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments from our YAML config.

    TrainingArguments is the HuggingFace abstraction over the training loop.
    It handles logging, checkpointing, LR scheduling, and device management
    so we don't have to write a manual training loop like in the receipt project.
    This is how production fine-tuning pipelines are structured.
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
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="epoch",           # run validation at the end of every epoch
        save_strategy="epoch",           # save a checkpoint at the end of every epoch
        load_best_model_at_end=train_cfg["save_best_only"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,         # lower eval_loss = better model
        bf16=True,                       # use bfloat16 for training on Apple Silicon
        report_to="none",                # disable wandb/tensorboard for now, keeps it simple
        remove_unused_columns=False,     # important: keep our custom 'labels' column
    )


def train():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "lora_config.yaml")
    config = load_config(config_path)

    # ── Step 1: Load model and tokenizer ──────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)

    # ── Step 2: Inject LoRA adapters, freeze base weights ─────────────────────
    model = apply_lora(model, config)

    # ── Step 3: Load and tokenize dataset ─────────────────────────────────────
    train_dataset, val_dataset = get_dataset(config_path, tokenizer)

    # ── Step 4: Data collator ──────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,   # pads sequence length to multiples of 8 for efficiency
        label_pad_token_id=-100, # ensures padding in labels is also masked from loss
    )

    # ── Step 5: Build trainer and run ─────────────────────────────────────────
    training_args = get_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("\n[train] Starting training...\n")
    trainer.train()

    # ── Step 6: Save the final LoRA adapter ───────────────────────────────────
    adapter_path = config["training"]["output_dir"]
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n[train] LoRA adapter saved to {adapter_path}")

    # Free memory after training
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    train()